import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { PredictionService } from './services/prediction.service';

interface PredictionResult {
  prediction: string;
  confidence: number;
  probability_yes: number;
  probability_no: number;
  detection_rate: number;
  risk_level: string;
}

interface ModelInfo {
  model_name: string;
  accuracy: number;
  features: string[];
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, HttpClientModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'Early Autism Detection System';
  
  formData = {
    A1_Score: 0,
    A2_Score: 0,
    A3_Score: 0,
    A4_Score: 0,
    A5_Score: 0,
    A6_Score: 0,
    A7_Score: 0,
    A8_Score: 0,
    A9_Score: 0,
    A10_Score: 0,
    age: 25,
    result: 0
  };

  predictionResult: PredictionResult | null = null;
  modelInfo: ModelInfo | null = null;
  loading = false;
  error: string | null = null;

  screeningQuestions = [
    { id: 'A1_Score', label: 'Often notices small sounds when others do not?' },
    { id: 'A2_Score', label: 'Usually concentrates more on the whole picture, rather than the small details?' },
    { id: 'A3_Score', label: 'Does not usually notice small changes in a set routine at home or work?' },
    { id: 'A4_Score', label: 'Finds it easy to imagine things?' },
    { id: 'A5_Score', label: 'Often has to go over things many times before understanding them?' },
    { id: 'A6_Score', label: 'Is fascinated by numbers or patterns?' },
    { id: 'A7_Score', label: 'Often does not know what to do in a social situation?' },
    { id: 'A8_Score', label: 'Finds it difficult to work out what other people are thinking or feeling?' },
    { id: 'A9_Score', label: 'Does not usually enjoy social situations?' },
    { id: 'A10_Score', label: 'Finds it hard to make new friends?' }
  ];

  constructor(private predictionService: PredictionService) {}

  ngOnInit() {
    this.loadModelInfo();
  }

  loadModelInfo() {
    this.predictionService.getModelInfo().subscribe({
      next: (data) => {
        this.modelInfo = data;
      },
      error: (err) => {
        console.error('[Frontend] Error loading model info:', err);
      }
    });
  }

  onPredict() {
    this.loading = true;
    this.error = null;
    this.predictionResult = null;

    // compute total score if not provided
    if (this.formData.result === 0) {
      this.formData.result = this.getTotalScore();
    }

    this.predictionService.predict(this.formData).subscribe({
      next: (result) => {
        this.predictionResult = result;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Error making prediction. Please try again.';
        console.error('[Frontend] Prediction error:', err);
        this.loading = false;
      }
    });
  }

  getTotalScore(): number {
    return this.screeningQuestions.reduce((sum, q) => sum + (this.formData as any)[q.id], 0);
  }

  resetForm() {
    this.formData = {
      A1_Score: 0, A2_Score: 0, A3_Score: 0, A4_Score: 0, A5_Score: 0,
      A6_Score: 0, A7_Score: 0, A8_Score: 0, A9_Score: 0, A10_Score: 0,
      age: 25, result: 0
    };
    this.predictionResult = null;
    this.error = null;
  }

  getRiskColor(riskLevel: string): string {
    switch(riskLevel) {
      case 'High Risk': return '#ef4444';
      case 'Medium Risk': return '#f97316';
      case 'Low Risk': return '#eab308';
      case 'Very Low Risk': return '#22c55e';
      default: return '#64748b';
    }
  }

  getConfidencePercentage(confidence: number): string {
    return (confidence * 100).toFixed(2);
  }
}
