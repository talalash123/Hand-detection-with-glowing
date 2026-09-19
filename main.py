import cv2
import numpy as np
import random
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands

# 1. MediaPipe Hand Tracking Setup
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index Finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle Finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring Finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm Base Connections
    (5, 9), (9, 13), (13, 17)
]

# 2. Fire Particle System Data Structure
particles = []

def create_fire_particle(x, y):
    """ Hand joint ke surrounding se floating fire particles generate karega """
    return {
        'x': x + random.randint(-4, 4),
        'y': y + random.randint(-4, 4),
        'vx': random.uniform(-1.2, 1.2),        # Horizontal float velocity
        'vy': random.uniform(-3.5, -1.0),       # Upward rising velocity (Fire buoyancy)
        'radius': random.randint(3, 7),
        'life': 1.0,                            # Particle opacity lifetime (1.0 to 0.0)
        'decay': random.uniform(0.04, 0.08)
    }

cap = cv2.VideoCapture(0)

print("Starting Fire & Glow Hand Tracker... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    glow_mask = np.zeros((h, w, 3), dtype=np.uint8)
    line_mask = np.zeros((h, w, 3), dtype=np.uint8)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            points = []
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))

            # Hand Joints Skeleton Lines (Fiery Orange/Cyan Overlay)
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                p1, p2 = points[p1_idx], points[p2_idx]
                cv2.line(glow_mask, p1, p2, (0, 120, 255), 8)    # Deep Orange-Fire Glow Base
                cv2.line(line_mask, p1, p2, (200, 255, 255), 2)  # Bright Hot Core Line

            # Har joint Node par Fire Particles Spawning
            for pt in points:
                cv2.circle(glow_mask, pt, 6, (0, 180, 255), -1)
                cv2.circle(line_mask, pt, 4, (255, 255, 255), -1)
                
                # Dynamic particle density
                if random.random() < 0.6:
                    particles.append(create_fire_particle(pt[0], pt[1]))

    # 3. Fire Physics Engine Simulation Update
    updated_particles = []
    for p in particles:
        p['x'] += p['vx']
        p['y'] += p['vy']
        p['life'] -= p['decay']
        p['radius'] = max(1, int(p['radius'] * 0.92))

        if p['life'] > 0 and p['radius'] > 0:
            px, py = int(p['x']), int(p['y'])
            if 0 <= px < w and 0 <= py < h:
                # Color transition: Yellow Flame Core -> Orange -> Reddish Edge
                if p['life'] > 0.6:
                    color = (50, 220, 255)   # Yellow-Bright
                elif p['life'] > 0.3:
                    color = (0, 140, 255)    # Flame Orange
                else:
                    color = (0, 30, 200)     # Dark Red Decay

                # Draw particles on masks
                cv2.circle(glow_mask, (px, py), p['radius'] + 2, color, -1)
                cv2.circle(line_mask, (px, py), p['radius'], (255, 255, 255), -1)
                
                updated_particles.append(p)

    particles = updated_particles

    # 4. Soft Fire Atmosphere Blur
    glow_blur = cv2.GaussianBlur(glow_mask, (25, 25), 0)

    # Multi-pass blending
    output = cv2.addWeighted(frame, 1.0, glow_blur, 1.6, 0)
    output = cv2.addWeighted(output, 1.0, line_mask, 0.9, 0)

    cv2.imshow("Fire & Glowing Particles Hand Tracker", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()