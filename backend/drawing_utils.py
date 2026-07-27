import cv2

# ---------------------------------------------------
# MediaPipe Hand Connections
# ---------------------------------------------------

HAND_CONNECTIONS = [

    (0,1),(1,2),(2,3),(3,4),

    (0,5),(5,6),(6,7),(7,8),

    (5,9),(9,10),(10,11),(11,12),

    (9,13),(13,14),(14,15),(15,16),

    (13,17),(17,18),(18,19),(19,20),

    (0,17)

]


# ---------------------------------------------------
# Draw Hand Landmarks
# ---------------------------------------------------

def draw_landmarks(frame, hand_landmarks):

    h, w = frame.shape[:2]

    points = []

    for lm in hand_landmarks:

        x = int(lm.x * w)
        y = int(lm.y * h)

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            4,
            (0,255,0),
            -1
        )

    for start, end in HAND_CONNECTIONS:

        if start < len(points) and end < len(points):

            cv2.line(
                frame,
                points[start],
                points[end],
                (255,0,0),
                2
            )

    return frame


# ---------------------------------------------------
# Bounding Box + Prediction
# ---------------------------------------------------

def draw_bounding_box_and_label(
    frame,
    bbox,
    prediction,
    confidence
):

    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0,255,0),
        2
    )

    label = prediction

    if confidence is not None:

        label += f" ({confidence:.1f}%)"

    (tw, th), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        2
    )

    cv2.rectangle(
        frame,
        (x1, y1 - 35),
        (x1 + tw + 10, y1),
        (0,255,0),
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1 + 5, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,0,0),
        2
    )

    return frame