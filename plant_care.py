import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import PyPDF2
from langchain.llms import GooglePalm

# Google PaLM API Key
apikey = st.secrets[apikey]  # Replace with your actual Google PaLM API key

# Initialize the PaLM LLM with LangChain
llm = GooglePalm(google_api_key=apikey, temperature=0.2)

# Function to extract text from the PDF hosted on GitHub
def extract_pdf_text(pdf_url):
    response = requests.get(pdf_url)
    response.raise_for_status()  # Check if the request was successful
    pdf_reader = PyPDF2.PdfReader(BytesIO(response.content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to find relevant information about the plant in the PDF text
def find_plant_info(plant_name, pdf_text):
    if plant_name.lower() in pdf_text.lower():
        start = pdf_text.lower().find(plant_name.lower())
        end = pdf_text.find("\n", start + 500)
        return pdf_text[start:end].strip()
    return "No specific information found for this plant."

# Function to generate plant care information using the LLM
def generate_plant_care_info(plant_name, pdf_text):
    plant_info = find_plant_info(plant_name, pdf_text)
    prompt = f"I have the following information about the plant {plant_name}:\n\n{plant_info}\n\nPlease provide detailed care instructions based on this information."
    response = llm(prompt)
    return response

# Set up the Streamlit app title and description
st.title("Plant Image Classification and Care Tips")
st.write("Upload an image to identify the plant and get care tips based on PDF information.")

# Load the model and the image processor
model_name = "umutbozdag/plant-identity"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Upload the PDF from GitHub
pdf_url = "https://www.kellogggarden.com/wp-content/uploads/2020/05/Monthly-Flower-Gardening-Guide.pdf"  # Replace with your actual PDF URL
pdf_text = extract_pdf_text(pdf_url)

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Move model and inputs to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted class index
    predicted_class_idx = logits.argmax(-1).item()

    # Access the labels directly from the model
    labels = model.config.id2label
    predicted_label = labels[predicted_class_idx]

    # Display the result
    st.write(f"**Predicted Label:** {predicted_label}")

    # Generate plant care tips using the LLM and PDF
    care_info = generate_plant_care_info(predicted_label, pdf_text)
    st.write(f"**Plant Care Tips:**\n{care_info}")

    # Display the image with the predicted label using matplotlib
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(f"Predicted Label: {predicted_label}", fontsize=16)
    ax.axis('off')
    st.pyplot(fig)
