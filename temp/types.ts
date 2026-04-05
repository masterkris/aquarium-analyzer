/**
 * TODO: We should probably add typescript to the frontend. Ignoring
 * for midpoint demo but modeled the response data here.
 */

export interface IAnalyzeResponse {
  total_fish: number;
  detections: Array<IFormattedDetection>;
  species_counts: Array<ISpeciesCount>;
}

interface IFormattedDetection {
  species: string;
  confidence: number;
}

interface ISpeciesCount {
  [species: string]: number;
}
