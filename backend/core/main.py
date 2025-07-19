from predict import Predict
from camrea import Camera
from time import sleep
from fastapi import FastAPI, WebSocket
from contextlib import asynccontextmanager
import logging
from typing import List, Tuple, Dict, Set
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predicter = Predict()
cam1 = Camera(0)
cam2 = Camera(1)

def compare_and_validate_scores(scores1: List[Tuple[int, str]], scores2: List[Tuple[int, str]], 
                               confidence_threshold: float = 0.6) -> List[Tuple[int, str]]:
    """
    Compare scores from two cameras and return the most reliable results.
    
    Args:
        scores1: Scores from camera 1
        scores2: Scores from camera 2 
        confidence_threshold: Minimum agreement ratio needed
    
    Returns:
        List of validated dart scores
    """
    if not scores1 and not scores2:
        return []
    
    # If only one camera has results, use those (but log it)
    if not scores1:
        logger.warning("Camera 1 detected no darts, using Camera 2 results")
        return scores2
    
    if not scores2:
        logger.warning("Camera 2 detected no darts, using Camera 1 results")
        return scores1
    
    # Both cameras have results - compare them
    logger.info(f"Camera 1 detected: {scores1}")
    logger.info(f"Camera 2 detected: {scores2}")
    
    # Strategy 1: Exact matches first
    exact_matches = []
    scores1_set = set(scores1)
    scores2_set = set(scores2)
    
    # Find exact matches
    for score in scores1_set.intersection(scores2_set):
        exact_matches.append(score)
    
    # Strategy 2: Score value matches (same points, different description)
    score_matches = []
    scores1_values = [s[0] for s in scores1]
    scores2_values = [s[0] for s in scores2]
    
    for i, score1 in enumerate(scores1):
        for j, score2 in enumerate(scores2):
            if (score1[0] == score2[0] and 
                score1 not in exact_matches and 
                score2 not in exact_matches):
                # Prefer the more specific description
                better_desc = score1[1] if len(score1[1]) > len(score2[1]) else score2[1]
                score_matches.append((score1[0], better_desc))
                break
    
    # Combine results
    validated_scores = exact_matches + score_matches
    
    # Strategy 3: If we have significant disagreement, use majority camera or apply logic
    total_detected = len(scores1) + len(scores2)
    agreement_ratio = len(validated_scores) / max(total_detected, 1)
    
    if agreement_ratio < confidence_threshold:
        logger.warning(f"Low agreement between cameras ({agreement_ratio:.2f}). Using best guess.")
        
        # Use the camera with more detections, or average the results
        if len(scores1) >= len(scores2):
            logger.info("Using Camera 1 results due to more detections")
            return scores1[:3]  # Max 3 darts
        else:
            logger.info("Using Camera 2 results due to more detections")
            return scores2[:3]  # Max 3 darts
    
    logger.info(f"Validation successful with {agreement_ratio:.2f} agreement")
    return validated_scores[:3]  # Max 3 darts per turn

def filter_reasonable_scores(scores: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """
    Filter out obviously incorrect scores based on dart rules.
    """
    valid_scores = []
    
    for score, desc in scores:
        # Basic validation
        if score < 0 or score > 180:  # Max possible single dart is 60 (T20)
            logger.warning(f"Filtering out impossible score: {score}")
            continue
            
        # Check for reasonable score ranges
        if score > 60 and not any(x in desc.upper() for x in ['T', 'TRIPLE']):
            logger.warning(f"High score {score} without triple indication: {desc}")
            # Still include it but log the concern
            
        valid_scores.append((score, desc))
    
    return valid_scores

def get_dart_count_from_cameras() -> Tuple[int, int]:
    """
    Get the current number of darts detected by each camera.
    This helps determine when darts are removed.
    """
    try:
        # You'll need to add a method to your Predict class to get dart count
        # without necessarily returning new dart scores
        count1 = predicter.get_dart_count(cam1.cam_num) if hasattr(predicter, 'get_dart_count') else 0
        count2 = predicter.get_dart_count(cam2.cam_num) if hasattr(predicter, 'get_dart_count') else 0
        return count1, count2
    except Exception as e:
        logger.error(f"Error getting dart counts: {e}")
        return 0, 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting dart detection system...")
    
    # Calibrate both cameras
    logger.info("Calibrating Camera 1...")
    calibrated1 = predicter.calabrate(cam1)
    if not calibrated1:
        logger.warning("Camera 1 calibration failed, retrying...")
        sleep(2)
        calibrated1 = predicter.calabrate(cam1)
    
    logger.info("Calibrating Camera 2...")
    calibrated2 = predicter.calabrate(cam2)
    if not calibrated2:
        logger.warning("Camera 2 calibration failed, retrying...")
        sleep(2)
        calibrated2 = predicter.calabrate(cam2)
    
    if not calibrated1 and not calibrated2:
        logger.error("Both cameras failed to calibrate!")
    elif not calibrated1:
        logger.warning("Only Camera 2 calibrated - using single camera mode")
    elif not calibrated2:
        logger.warning("Only Camera 1 calibrated - using single camera mode")
    else:
        logger.info("Both cameras calibrated successfully!")
    
    yield
    
    logger.info("Shutting down dart detection system...")

app = FastAPI(lifespan=lifespan)

# Keep track of last known state to detect changes
last_validated_scores = []
consecutive_empty_readings = 0

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global last_validated_scores, consecutive_empty_readings
    
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                # Get scores from both cameras
                scores1 = predicter.run(cam1)
                scores2 = predicter.run(cam2)
                
                # Filter obviously wrong scores
                scores1_filtered = filter_reasonable_scores(scores1) if scores1 else []
                scores2_filtered = filter_reasonable_scores(scores2) if scores2 else []
                
                # Compare and validate
                validated_scores = compare_and_validate_scores(scores1_filtered, scores2_filtered)
                
                # Get dart counts for removal detection
                count1, count2 = get_dart_count_from_cameras()
                max_dart_count = max(count1, count2)
                
                # Detect dart removal (when we had darts but now we don't)
                if last_validated_scores and not validated_scores and max_dart_count == 0:
                    consecutive_empty_readings += 1
                    if consecutive_empty_readings >= 2:  # Confirm removal over multiple readings
                        logger.info("Darts removed from board")
                        validated_scores = []  # Explicitly send empty to indicate removal
                        consecutive_empty_readings = 0
                    else:
                        # Keep last scores to avoid flickering
                        validated_scores = last_validated_scores
                else:
                    consecutive_empty_readings = 0
                
                # Update last known state
                if validated_scores:
                    last_validated_scores = validated_scores
                
                # Send response
                response = {
                    "scores": validated_scores,
                    "dart_count": max_dart_count,
                    "camera1_count": count1,
                    "camera2_count": count2,
                    "camera1_scores": scores1_filtered,
                    "camera2_scores": scores2_filtered,
                    "validation_method": "dual_camera_comparison"
                }
                
                await websocket.send_json(response)
                
            except Exception as e:
                logger.error(f"Error processing camera data: {e}")
                # Send error response
                await websocket.send_json({
                    "scores": [],
                    "dart_count": 0,
                    "error": str(e)
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket client disconnected")
