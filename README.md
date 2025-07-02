# Cat-and-Dog-Image-Classifier

# 🐶🐱 Cat and Dog Image Classifier using ViT (Vision Transformer)

This project is a Flask web application that uses Hugging Face's **ViT (Vision Transformer)** model to classify uploaded images as **Cat**, **Dog**, or **Invalid** (neither cat nor dog). It utilizes the pre-trained `google/vit-base-patch16-224` model from the Transformers library.

## Features

- Upload any image via the web interface.
- Automatically classifies the image using Vision Transformer.
- Detects if the image contains a cat, dog, or something else.
- Highlights valid animal classes (Cat or Dog) based on ImageNet labels.

## Model Info

- **Model**: `google/vit-base-patch16-224`
- **Source**: [Hugging Face](https://huggingface.co/google/vit-base-patch16-224)
- **Framework**: PyTorch

## Installation


# Clone the repository
git clone https://github.com/your-username/cat-dog-vit-classifier.git
cd cat-dog-vit-classifier

# Create virtual environment
python -m venv venv
# Activate it
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py

# Then open in your browser:
# http://127.0.0.1:5000/
`

# requirements.txt


flask
torch
transformers
pillow


## Folder Structure


cat-dog-vit-classifier/
├── app.py
├── templates/
│   └── index.html
├── uploads/
│   └── [uploaded images]
├── static/
│   └── [optional static files]
├── requirements.txt
└── README.md


## How It Works

* Upload an image through the web interface.
* The model processes the image and predicts a label.
* If the label contains "cat", it returns `Cat (Valid)`.
* If the label contains known dog breeds, it returns `Dog (Valid)`.
* Otherwise, it returns `Invalid (Not a Cat or Dog)`.

### Prediction Logic

python
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits
predicted_idx = logits.argmax(-1).item()
label = model.config.id2label[predicted_idx]


## License

This project is open-source and available under the **MIT License**.

## Author

**Apoorva Mahendar M**

## Acknowledgements

* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [PyTorch](https://pytorch.org/)
* [Flask](https://flask.palletsprojects.com/)




