export interface IFileUploadForm {
  tankSize: number;
  selectFile: FileList;
}

export interface IAnalyzeResponse {
  total_fish: number;
  detections: Array<IFormattedDetection>;
  species_counts: Record<string, number>;
  recommendations: string;
}

export interface IFormattedDetection {
  species: string;
  confidence: number;
}
