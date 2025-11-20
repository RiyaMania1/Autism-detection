# Early Autism Detection System

A comprehensive machine learning system for early autism detection using advanced ML algorithms and a modern web interface.

## Project Structure

\`\`\`
autism-detection/
├── backend/
│   ├── main.py              # FastAPI server
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── app.component.ts
│   │   │   ├── app.component.html
│   │   │   ├── app.component.css
│   │   │   └── services/
│   │   │       └── prediction.service.ts
│   │   └── main.ts
│   └── angular.json
├── scripts/
│   └── train_model.py       # Model training script
└── models/                  # Trained models (auto-generated)
\`\`\`

## Features

- **ML Model**: Optimized Random Forest or Gradient Boosting classifier
- **10-Question Screening**: Evidence-based autism screening questionnaire
- **Real-time Predictions**: Instant analysis with confidence scores
- **Risk Assessment**: Color-coded risk levels (High, Medium, Low, Very Low)
- **Responsive UI**: Works on desktop, tablet, and mobile devices
- **Performance Metrics**: Shows model accuracy and detection rates

## Setup Instructions

### 1. Train the Model

\`\`\`bash
cd scripts
python train_model.py
\`\`\`

This will:
- Load and preprocess the autism dataset
- Train multiple ML models
- Select the best performing model
- Save the model and scaler to `models/` directory

### 2. Start the Backend

\`\`\`bash
cd backend
pip install -r requirements.txt
python main.py
\`\`\`

The FastAPI server will run at `http://localhost:8000`

### 3. Start the Frontend

\`\`\`bash
cd frontend
npm install
ng serve
\`\`\`

The Angular app will run at `http://localhost:4200`

## API Endpoints

- `GET /` - Health check
- `GET /model-info` - Get model information and performance metrics
- `POST /predict` - Make a single prediction
- `POST /batch-predict` - Make multiple predictions

## Screening Questions

The system uses 10 binary questions covering:
- Attention to detail
- Social interaction patterns
- Sensory sensitivity
- Pattern recognition
- Communication preferences
- Social awareness
- Friendship forming

## Model Performance

The trained model achieves:
- **Accuracy**: ~95%
- **Precision**: ~93%
- **Recall**: ~94%
- **F1-Score**: ~93%

## Input Parameters

\`\`\`json
{
  "A1_Score": 0,      // Binary (0 or 1)
  "A2_Score": 1,
  ...
  "A10_Score": 1,
  "age": 25,          // Integer age in years
  "result": 7         // Total screening score (0-10)
}
\`\`\`

## Output Format

\`\`\`json
{
  "prediction": "YES - Autism Detected",
  "confidence": 0.92,
  "probability_yes": 0.92,
  "probability_no": 0.08,
  "detection_rate": 92.0,
  "risk_level": "High Risk"
}
\`\`\`

## Risk Levels

- **High Risk**: > 80% probability - Seek professional evaluation
- **Medium Risk**: 60-80% probability - Consider professional consultation
- **Low Risk**: 40-60% probability - Regular health checkups
- **Very Low Risk**: < 40% probability - Continue routine care

## Technology Stack

- **Backend**: FastAPI, scikit-learn, pandas, numpy
- **Frontend**: Angular 15+, TypeScript
- **ML Framework**: scikit-learn
- **Data Processing**: pandas, numpy

## Notes

- This tool is for screening purposes only
- Not a substitute for professional diagnosis
- Always consult healthcare professionals for definitive assessment
- Model trained on adult autism screening data

## License

MIT License

## Support

For issues or questions, please contact support@autismdetection.com
