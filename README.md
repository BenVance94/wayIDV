# wayIDV
Who Are You (WAY) is a Python-based Fraud Prevention Tool designed to detect and analyze fake IDs. It leverages advanced image processing and OCR techniques to identify potential fraud and validate ID authenticity.

## Features

- **Image Analysis**:
  - Detects fake IDs using multiple indicators
  - Analyzes image quality and text placement
  - Identifies potential fraud based on image metrics

- **OCR Validation**:

    - Validates text against official ID templates
    - Checks for text consistency and placement
    - Verifies microprint and security features

- **Output**:

| Score/Indicator | Expected Values | Description | What it Means |
|----------------|-----------------|-------------|---------------|
| Fraud Score | 0-100 | Numerical score | The higher the number, the more likely the ID is fake. Think of it like a test score for fakeness:<br>• 0-30: Very likely genuine<br>• 31-60: Requires attention<br>• 61-100: Likely fraudulent |
| Risk Level | "Low", "Medium", "High" | Overall risk assessment | A simple way to understand the risk:<br>• Low: Likely genuine<br>• Medium: Some concerns<br>• High: Likely fraudulent |
| Component Scores | Text: 0-100<br>Image: 0-100 | Separate scores for text and image | Shows which specific aspects of the ID raised concerns:<br>• Text Score: How suspicious the text appears<br>• Image Score: How suspicious the image quality is |
| Quality Metrics | DPI: 300-1200<br>Brightness: 0-100<br>Contrast: 0-100<br>Sharpness: 0-100 | Technical measurements | How clear and well-produced the ID is compared to official standards. Higher values generally indicate better quality. |
| Fake Indicators | Examples:<br>• "Invalid font"<br>• "Missing hologram"<br>• "Incorrect UV pattern"<br>• "Misaligned text"<br>• "Invalid barcode" | List of suspicious elements | Specific problems found that suggest the ID might be fake. Each indicator includes a detailed explanation of the issue. |
| Extracted Text | Fields like:<br>• Full Name<br>• Date of Birth<br>• ID Number<br>• Address<br>• Issue Date<br>• Expiry Date | Text found on the ID | All text read from the ID, making it easy to verify against provided information. Format varies by ID type and jurisdiction. |

## Installation


## Adding a Driver's License Image

1. Go to the `images` folder and add your image there.
2. Edit the `input.json` file to include the details of the image you added.
3. Save and run the tool again.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
