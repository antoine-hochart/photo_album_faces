import os
import platform
import cv2

from time import time
from PIL import Image    

from face_cluster import load_face_embeddings

from pkg.facenet import get_face_embedding
from pkg.gdl import GDL
from pkg.mtcnn import MTCNN, extract_face
from pkg.pseudo import generate_pseudos

######################################################################

def box_match(inner_box, outer_box, margin=100):
    x0, y0, x1, y1 = inner_box
    xa, ya, xb, yb = outer_box
    match = True
    if x0 < xa-margin or y0 < ya-margin or x1 > xb+margin or y1 > yb+margin:
        match = False
    return match


def recognize_faces(algo, strides=(1, 1), margin=0):
    mtcnn = MTCNN(select_largest=False, keep_all=True, device='cpu').eval()

    pseudos = generate_pseudos(len(gdl.clusters))
    pseudos += ['???']

    if platform.system() == 'Windows':
        cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # initialization
    n_detect, n_recog = 0, 0
    boxes = None
    labels, label_boxes = [], []
    gif_toggle = 0
    frames = []

    while True:
        ret, frame = cap.read()
        cv2.putText(
            img=frame,
            text="quit: q",
            org=(10, frame_height-10),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1.5,
            color=(0, 0, 0),
            thickness=1
            )
        # detect every strides[0] frames
        if n_detect == 0:
            boxes, probs = mtcnn.detect(frame, landmarks=False)
        # recognize every strides[1] frames
        if n_recog == 0 and boxes is not None:
            faces = mtcnn.extract(frame, boxes, save_path=None)
            embeddings = get_face_embedding(faces, device='cpu')
            labels = algo.predict(embeddings, alpha=0.50)
            label_boxes = boxes.copy()
        # draw boxes and labels
        if boxes is not None:
            # draw boxes
            for box, prob in zip(boxes, probs):
                if prob > 0.95:
                    x0, y0, x1, y1 = box.astype('int')
                    cv2.rectangle(frame, (x0-margin, y0-margin), (x1+margin, y1+margin),
                                  (0, 255, 0), 2)
            # display labels
            for box, label_box, label, prob in zip(boxes, label_boxes, labels, probs):
                if prob > 0.95 and box_match(box, label_box):
                    x0, y0, x1, y1 = box.astype('int')
                    cv2.putText(
                        img=frame,
                        text=pseudos[label],
                        org=(x0-margin, y0-margin-5),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1.5,
                        color=(0, 255, 0),
                        thickness=2
                        )
        # update frame no.
        n_detect  = (n_detect + 1) % strides[0]
        n_recog  = (n_recog + 1) % strides[1]
        # display the resulting frame
        cv2.imshow('video', frame)
        # keyboard command
        key = cv2.waitKey(1)
        if key == ord(' '):
            # toggle frame saving option
            gif_toggle = 1 - gif_toggle
            print("{} saving frames".format('Start' if gif_toggle else 'End'))
        elif key == ord('q'):
            # keyboard interrupt
            break

        if gif_toggle:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img))

    cap.release()
    cv2.destroyAllWindows()

    if frames:
        gif = [img.resize((480, 360), Image.ANTIALIAS) for img in frames]
        img_dir = os.path.join(os.path.dirname(__file__), '..', 'imgs')
        os.makedirs(os.path.abspath(img_dir), exist_ok=True)
        img_path = os.path.join(img_dir, 'recog.gif')
        gif[0].save(img_path, save_all=True, optimize=True,
                    append_images=gif[1::2], loop=0)


######################################################################

if __name__ == "__main__":
    print("Face embedding")
    t0 = time()
    faces, fnames = load_face_embeddings()
    print("Done ({:.2f}s)".format(time() - t0))

    print()
    print("Face clustering")
    t0 = time()
    gdl = GDL(n_neighbors=5, eps=0.050)
    gdl.fit(faces)
    print("Done ({:.2f}s)".format(time() - t0))

    print()
    print("Face recognition")
    recognize_faces(gdl, strides=(5, 25), margin=5)
