import re
import unicodedata
from typing import Dict, List, Set, Tuple
import logging

logger = logging.getLogger(__name__)

class CropNormalizer:
    def __init__(self):
        self.crop_mappings = {
            
            'aceitunas': 'aceituna',
            'aceituna': 'aceituna',
            'fresas': 'fresa',
            'fresa': 'fresa',
            'manzanas': 'manzana',
            'manzana': 'manzana',
            'alubias': 'alubia',
            'alubia': 'alubia',
            'avellanas': 'avellana',
            'avellana': 'avellana',
            'ciruelas': 'ciruela',
            'ciruela': 'ciruela',
            'grosellas': 'grosella',
            'grosella': 'grosella',
            'higos': 'higo',
            'higo': 'higo',
            'peras': 'pera',
            'pera': 'pera',
            'tomates': 'tomate',
            'tomate': 'tomate',
            'uvas': 'uva',
            'uva': 'uva',
            'zanahorias': 'zanahoria',
            'zanahoria': 'zanahoria',
            
            
            'melocoton': 'melocotón',
            'melocotón': 'melocotón',
            'perejil': 'perejil',
            
          
            'vid': 'uva',
            'albaricoque': 'albaricoque',
            'cebada': 'cebada',
            'lentejas': 'lenteja',
            'lenteja': 'lenteja',
            'maiz': 'maíz',
            'maíz': 'maíz',
            'naranjas': 'naranja',
            'naranja': 'naranja',
            'soja': 'soja',
            'trigo': 'trigo',
            'albahaca': 'albahaca'
        }
        
     
        self.reverse_mappings = {}
        for original, normalized in self.crop_mappings.items():
            if normalized not in self.reverse_mappings:
                self.reverse_mappings[normalized] = []
            self.reverse_mappings[normalized].append(original)
    
    def normalize_crop_name(self, crop_name: str) -> str:
    
        if not crop_name:
            return crop_name
            
        try:
          
            normalized = crop_name.lower().strip()
            
          
            normalized = self._remove_accents(normalized)
            
           
            if normalized in self.crop_mappings:
                return self.crop_mappings[normalized]
            
        
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing crop name '{crop_name}': {e}")
            return crop_name
    
    def _remove_accents(self, text: str) -> str:
    
        try:
           
            normalized = unicodedata.normalize('NFD', text)
           
            without_accents = ''.join(c for c in normalized if not unicodedata.combining(c))
            return without_accents
        except Exception as e:
            logger.error(f"Error removing accents from '{text}': {e}")
            return text
    
    def get_all_variations(self, normalized_name: str) -> List[str]:
       
        try:
            if normalized_name in self.reverse_mappings:
                return self.reverse_mappings[normalized_name]
            return [normalized_name]
        except Exception as e:
            logger.error(f"Error getting variations for '{normalized_name}': {e}")
            return [normalized_name]
    
    def normalize_dataset(self, crops_data: List[Dict]) -> Tuple[List[Dict], Dict[str, str]]:
       
        try:
            normalized_data = []
            normalization_log = {}
            
            for record in crops_data:
                if 'tipo_de_cultivo' in record:
                    original_name = record['tipo_de_cultivo']
                    normalized_name = self.normalize_crop_name(original_name)
                    
                  
                    normalized_record = record.copy()
                    normalized_record['tipo_de_cultivo'] = normalized_name
                    normalized_data.append(normalized_record)
                    
                  
                    if original_name != normalized_name:
                        normalization_log[original_name] = normalized_name
                        logger.info(f"Normalized '{original_name}' -> '{normalized_name}'")
                else:
                    normalized_data.append(record)
            
            return normalized_data, normalization_log
            
        except Exception as e:
            logger.error(f"Error normalizing dataset: {e}")
            return crops_data, {}
    
    def get_unique_normalized_crops(self, crops_data: List[Dict]) -> Set[str]:
       
        try:
            unique_crops = set()
            for record in crops_data:
                if 'tipo_de_cultivo' in record:
                    normalized = self.normalize_crop_name(record['tipo_de_cultivo'])
                    unique_crops.add(normalized)
            return unique_crops
        except Exception as e:
            logger.error(f"Error getting unique normalized crops: {e}")
            return set()
    
    def suggest_corrections(self, crop_name: str) -> List[str]:
     
        try:
            normalized = self.normalize_crop_name(crop_name)
            suggestions = []
            
            
            for original, normalized_mapped in self.crop_mappings.items():
                if normalized_mapped == normalized and original != crop_name.lower():
                    suggestions.append(original)
            
            return suggestions[:5] 
            
        except Exception as e:
            logger.error(f"Error suggesting corrections for '{crop_name}': {e}")
            return []
    
    def validate_crop_name(self, crop_name: str) -> Dict[str, any]:
       
        try:
            if not crop_name or len(crop_name.strip()) < 2:
                return {
                    "is_valid": False,
                    "error": "El nombre del cultivo debe tener al menos 2 caracteres",
                    "suggestions": []
                }
            
            normalized = self.normalize_crop_name(crop_name)
            original_lower = crop_name.lower().strip()
            
         
            if original_lower != normalized:
                suggestions = self.suggest_corrections(crop_name)
                return {
                    "is_valid": True,
                    "needs_normalization": True,
                    "original": crop_name,
                    "normalized": normalized,
                    "suggestions": suggestions,
                    "message": f"El nombre se normalizará a '{normalized}'"
                }
            else:
                return {
                    "is_valid": True,
                    "needs_normalization": False,
                    "original": crop_name,
                    "normalized": normalized,
                    "suggestions": [],
                    "message": "El nombre está correctamente normalizado"
                }
                
        except Exception as e:
            logger.error(f"Error validating crop name '{crop_name}': {e}")
            return {
                "is_valid": False,
                "error": f"Error validando el nombre: {str(e)}",
                "suggestions": []
            } 