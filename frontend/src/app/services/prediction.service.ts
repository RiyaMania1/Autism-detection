import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface PredictionRequest {
  A1_Score: number;
  A2_Score: number;
  A3_Score: number;
  A4_Score: number;
  A5_Score: number;
  A6_Score: number;
  A7_Score: number;
  A8_Score: number;
  A9_Score: number;
  A10_Score: number;
  age: number;
  result: number;
}

export interface PredictionResponse {
  prediction: string;
  confidence: number;
  probability_yes: number;
  probability_no: number;
  detection_rate: number;
  risk_level: string;
}

@Injectable({ providedIn: 'root' })
export class PredictionService {
  // backend API
  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  // POST request to /predict
  predict(data: PredictionRequest): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>(`${this.apiUrl}/predict`, data);
  }

  // GET request to /model-info
  getModelInfo(): Observable<any> {
    return this.http.get(`${this.apiUrl}/model-info`);
  }

  // health check
  healthCheck(): Observable<any> {
    return this.http.get(`${this.apiUrl}/`);
  }
}
