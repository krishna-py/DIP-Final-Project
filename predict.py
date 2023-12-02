import torch
from torchvision.transforms import ToTensor
from PIL import Image

# Load the pretrained model
model = YourModelClass.load_from_checkpoint('path_to_checkpoint')

# Set the model to evaluation mode
model.eval()

# Load the image
image = Image.open('path_to_image')
image = ToTensor()(image).unsqueeze(0)  # Convert image to tensor and add batch dimension

# Run predictions per pixel
with torch.no_grad():
    output = model(image)

# Process the output as per your requirements
# ...

# Example: Get the predicted class for each pixel
predicted_classes = torch.argmax(output, dim=1)

# Example: Convert the predicted classes to numpy array
predicted_classes = predicted_classes.squeeze().cpu().numpy()

# Example: Display the predicted classes
print(predicted_classes)
