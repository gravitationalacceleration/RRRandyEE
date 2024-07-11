import cv2
import numpy as np
import tensorflow as tf
import time

#전체 메커니즘
# 영상 촬영 -> 영상 속 모션 분석 -> 분석한 결과
#포함한 화면 송출 to mqtt server


#내가 할 거는 저기서 영상 촬영해서 모션 분석하기.

#카메라 오픈 앤 클로징

dt = 1

model_path = './movenet_thunder.tflite'

MJPEG_HOST = 'MJPEG HOST IP'
MJPEG_PORT = 1234
previous_keypoints = None
prevact = None

def preprocess_image(pil_img, input_size=256):
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (input_size, input_size))
    img = img.astype(np.uint8)
    img = np.expand_dims(img, axis=0)
    return img
 
def streaming():
    global originalimage
    originalimage = cv2.VideoCapture(0)
    originalimage.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    originalimage.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

    while True:
        ret, frame = originalimage.read()
        cv2.imshow("test", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    originalimage.release()
    cv2.destroyAllWindows
    return originalimage
    
#사진을 numpy배열로 바꾸기
def numpyingimage ():
    n = np.array(streaming()) #캡쳐받은 영상을 numpy배열로 변환
    n = np.expand_dims(n, axis=0) #차원추가 (?)
    return n #numpy배열이 된 image 반환
#tensorflow를 통한 모델 교육을 위한 모델 생성
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path = model_path)
    interpreter.allocate_tensors()
    return interpreter
#속도에 대해 다루는 함수
def get_speed(previous_keypoints, current_keypoints,dir=0):
    if previous_keypoints is None:
        return 0

    if dir=='x': 
        previous_keypoints=previous_keypoints[:,0]
        current_keypoints=current_keypoints[:,1]
    elif dir=='y':
        previous_keypoints= previous_keypoints[:,0]
        current_keypoints= current_keypoints[:,1]

    ignore_indices = [0,1,2,3,4,7, 8, 9, 10]

    mask = np.ones(previous_keypoints.shape[0], dtype=bool)
    mask[ignore_indices] = False

    previous_keypoints = previous_keypoints[mask]
    current_keypoints = current_keypoints[mask]

    displacement = np.sqrt(np.sum((current_keypoints - previous_keypoints) ** 2, axis=1))
    mean_displacement = np.mean(displacement)

    return mean_displacement

def calculate_angle_between_lines(p1, p2, p3, p4):
    v1 = np.abs(np.array(p2) - np.array(p1))
    v2 = np.abs(np.array(p4) - np.array(p3))
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    angle = np.arccos(cosine)
    return np.degrees(angle)

def is_person_lying(keypoints):
    excluded_indices = [0, 1, 2, 3, 4, 7, 8, 9, 10]
    filtered_keypoints = np.delete(keypoints, excluded_indices, axis=0)

    x_points, y_points = filtered_keypoints[:, 0], filtered_keypoints[:, 1]

    x_min, x_max = x_points.min(), x_points.max()
    y_min, y_max = y_points.min(), y_points.max()
    width = x_max - x_min
    height = y_max - y_min

    ratio = width / height

    lying_threshold_ratio = 1.0

    eye_center = (keypoints[2] + keypoints[1]) / 2
    shoulder_center = (keypoints[5] + keypoints[6]) / 2
    hip_center = (keypoints[11] + keypoints[12]) / 2
    ankle_center = (keypoints[15] + keypoints[16]) / 2

    angle = calculate_angle_between_lines(shoulder_center, hip_center, hip_center, ankle_center)
    lying_threshold_angle = 30

    anglespeed = get_speed(keypoints, keypoints+1) / (eye_center - shoulder_center)
    omega = anglespeed.tolist()

    if ratio < lying_threshold_ratio and angle < lying_threshold_angle:
        for i in omega:
            if i < 0.3:
                return "lying"
                continue
            else:
                break
        return "collapse"

def classify_activity(keypoints, previous_keypoints, confi, prevact):
    global previous_speed
    threshold_height = 0.42
    speed = get_speed(previous_keypoints, keypoints)

    hcenter = (keypoints[11]+keypoints[12])/2
    acenter = (keypoints[15]+keypoints[16])/2

    center = acenter-hcenter
    hei = (keypoints[5]+keypoints[6])/2
    heicenter = acenter-hei
    center = heicenter/hcenter

    previous_speed = speed
    if is_person_lying(keypoints) == "Lying":
        return "Lying"
    elif is_person_lying(keypoints) == "collapse":
        return "Warning! collapsed"
    elif confi < 0.3:
        return "Unknown"
    elif center[0] < threshold_height:
        return "Sitting"
    elif speed > 0.013: #박사님 코드에서 speed_threshold
        if prevact == "Sitting":
            return "Standing"
        if speed > 0.050:
            return "Running"
        else:
            return "Walking"
    elif keypoints[9] == keypoints[10]:
        return "clapping"
    else:
        return "Standing"

def extract_keypoints_for_drawing(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    if keypoints.size == 0:
        raise ValueError("No keypoints detected")
    return keypoints

def inference_label():
    global interpreter, previous_keypoints, prevact
    cam = cv2.VideoCapture(0)
    while True:
        # url = f"http://{MJPEG_HOST}:{MJPEG_PORT}/image.jpg"
        # response = requests.get(url)
        # frame = response.content
        # aimage = Image.open(io.BytesIO(frame))

        ret, aimage = cam.read()
        
        img = preprocess_image(aimage)
        results = extract_keypoints_for_drawing(interpreter, img)
        confidence_scores = results[:, :, :, 2]
        confidence_scores = confidence_scores[0][0][5:]
        confidence_mean = np.mean(confidence_scores)
        keypoints = results[0][0][:,:2]

        activity = classify_activity(keypoints, previous_keypoints, confidence_mean, prevact)
        prevact = activity
        previous_keypoints = keypoints
        print(activity)
        
        frame = np.array(aimage)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #레이블링 결과를 실시간으로 확인
        cv2.putText(frame, f'Activity: {activity}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Activity Recognition', frame)

        motion = {"walking, sitting, standing"}

        for i in motion:

            if classify_activity(keypoints, previous_keypoints, confidence_mean, prevact) == i:
                if classify_activity(keypoints, previous_keypoints, confidence_mean, prevact) == "Warning! collapsed":
                    return "Warning! Collapsed"

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    global interpreter
    interpreter = load_model(model_path)
    a = inference_label()

    if a == "Warning! Collapsed":
        max_time = 300 #골든타임
        if time.time() == max_time:
            print("119")