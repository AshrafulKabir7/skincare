import os
import base64
import numpy as np
import cv2
from django.shortcuts import render
from django.conf import settings
from .forms import ImageUploadForm
from inference_sdk import InferenceHTTPClient

# Initialize the Inference client
CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key="QnNomR9ejQvCupdtNEDc"  # Replace with your actual API key
)

def home(request):
    return render(request, 'detection/home.html')

def detect_skin_problem(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_path = os.path.join(settings.MEDIA_ROOT, image.name)

            # Save the image temporarily
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            # Call the Roboflow API
            result = CLIENT.infer(image_path, model_id="skin-problem-multilabel/1")

            # Read the image using OpenCV from the saved path
            img = cv2.imread(image_path)

            # Annotate the image with the predicted classes and their confidence
            for class_name in result['predicted_classes']:
                confidence = result['predictions'][class_name]['confidence']
                cv2.putText(img, f'{class_name}: {confidence:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Convert the annotated image to base64 for rendering
            _, img_encoded = cv2.imencode('.jpg', img)
            img_b64 = base64.b64encode(img_encoded).decode('utf-8')

            # Delete the image after processing
            os.remove(image_path)

            return render(request, 'detection/result.html', {'result': result, 'image': img_b64})
    else:
        form = ImageUploadForm()

    return render(request, 'detection/upload.html', {'form': form})


