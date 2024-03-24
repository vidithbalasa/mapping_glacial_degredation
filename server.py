from flask import Flask, request, send_file
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from PIL import Image
import sys
import json

# Assuming the segment_anything directory is on the parent directory
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

app = Flask(__name__)
CORS(app)

# Initialize SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Helper functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

@app.route('/segment', methods=['POST'])
def segment_image():
    if 'image' not in request.files:
        return 'No image part', 400
    file = request.files['image']
    if file.filename == '':
        return 'No selected image', 400

    # Assume coordinates are sent as JSON in the format: {'x': 150, 'y': 250}
    if 'coordinates' not in request.form:
        return 'No coordinates provided', 400
    coordinates = json.loads(request.form['coordinates'])
    input_point = np.array([[coordinates['x'], coordinates['y']]])
    input_label = np.array([1])

    image = Image.open(file).convert('RGB')
    image_np = np.array(image)

    # Process the image with SAM
    predictor.set_image(image_np)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # Visualization
    mask = masks[scores.argmax()]
    score = scores.max()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np)
    show_mask(mask, ax)
    show_points(input_point, input_label, ax)
    ax.set_title(f"Score: {score:.3f}", fontsize=18)
    ax.axis('off')

    # Remove padding and margins around the image
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    # If the aspect ratio needs to be set
    ax.set_aspect('auto')

    # Convert plot to image and send as response
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
