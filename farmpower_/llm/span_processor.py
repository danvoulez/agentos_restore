"""
Span Processor for LogLineOS/DiamondSpan
Uses LLMs to process, validate, and enhance Diamond Spans
"""
import os
import json
import logging
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from llm.llm_service import LLMService, LLMConfig
from core.diamond_span import DiamondSpan

# Configure logging
logger = logging.getLogger("SpanProcessor")

class SpanProcessor:
    """
    Processor for Diamond Spans using LLMs
    """
    
    def __init__(self, llm_service: LLMService = None):
        self.llm = llm_service or LLMService(
            LLMConfig(model_id=os.getenv("LLM_MODEL_ID", "gpt-3.5-turbo"))
        )
        self.span_cache = {}  # Cache processed spans
        self.initialized = False
    
    async def initialize(self):
        """Initialize the processor"""
        if not self.initialized:
            await self.llm.initialize()
            self.initialized = True
    
    async def process_span(self, span: Union[DiamondSpan, Dict[str, Any]]) -> Dict[str, Any]:
        """Process a span with LLM to enhance, validate, or transform it"""
        # Ensure initialization
        if not self.initialized:
            await self.initialize()
        
        # Convert to dict if needed
        if isinstance(span, DiamondSpan):
            span_dict = {
                "id": span.id,
                "parent_ids": span.parent_ids,
                "content": span.content,
                "metadata": span.metadata
            }
        else:
            span_dict = span
        
        # Generate a cache key
        cache_key = hashlib.sha256(json.dumps(span_dict, sort_keys=True).encode()).hexdigest()
        
        # Check cache
        if cache_key in self.span_cache:
            return self.span_cache[cache_key]
        
        # Process based on span kind
        kind = span_dict.get("kind") or span_dict.get("content", {}).get("kind", "unknown")
        
        # Different processing based on span kind
        if kind == "diamond":
            result = await self._process_diamond_span(span_dict)
        elif kind in ["train", "fine_tune"]:
            result = await self._process_training_span(span_dict)
        elif kind in ["simulate", "predict"]:
            result = await self._process_prediction_span(span_dict)
        elif kind == "governance_policy":
            result = await self._process_governance_span(span_dict)
        else:
            # Default processing
            result = await self._process_generic_span(span_dict)
        
        # Cache the result
        self.span_cache[cache_key] = result
        
        return result
    
    async def validate_span(self, span: Union[DiamondSpan, Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a span for correctness, ethics and quality"""
        # Ensure initialization
        if not self.initialized:
            await self.initialize()
        
        # Convert to dict if needed
        if isinstance(span, DiamondSpan):
            span_dict = {
                "id": span.id,
                "parent_ids": span.parent_ids,
                "content": span.content,
                "metadata": span.metadata
            }
        else:
            span_dict = span
        
        # Prepare validation prompt
        kind = span_dict.get("kind") or span_dict.get("content", {}).get("kind", "unknown")
        
        prompt = f"""
        # DIAMOND SPAN VALIDATION
        
        Please validate the following Diamond Span:
        
        ```json
        {json.dumps(span_dict, indent=2)}
        ```
        
        Perform the following checks:
        1. Structural validity (required fields and format)
        2. Semantic coherence (does the content make sense)
        3. Ethical compliance (no harmful content)
        4. Quality assessment (value of the span)
        
        Provide your validation result as JSON with the following structure:
        {{
          "is_valid": true/false,
          "structural_score": 0-10,
          "coherence_score": 0-10,
          "ethical_score": 0-10,
          "quality_score": 0-10,
          "overall_score": 0-10,
          "issues": ["issue1", "issue2", ...],
          "recommendations": ["rec1", "rec2", ...]
        }}
        
        Only respond with the JSON.
        """
        
        # Generate validation
        validation_text, _ = await self.llm.generate_text(prompt)
        
        # Parse result
        try:
            validation = json.loads(validation_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the text
            import re
            json_match = re.search(r'({.*})', validation_text, re.DOTALL)
            if json_match:
                try:
                    validation = json.loads(json_match.group(1))
                except:
                    validation = {
                        "is_valid": False,
                        "structural_score": 0,
                        "coherence_score": 0,
                        "ethical_score": 0,
                        "quality_score": 0,
                        "overall_score": 0,
                        "issues": ["Failed to parse validation result"],
                        "recommendations": ["Check the span format"]
                    }
            else:
                validation = {
                    "is_valid": False,
                    "structural_score": 0,
                    "coherence_score": 0,
                    "ethical_score": 0,
                    "quality_score": 0,
                    "overall_score": 0,
                    "issues": ["Failed to parse validation result"],
                    "recommendations": ["Check the span format"]
                }
        
        logger.info(f"Validated span {span_dict.get('id', 'unknown')}: valid={validation['is_valid']}, score={validation.get('overall_score', 0)}")
        
        return validation
    
    async def enhance_span(self, span: Union[DiamondSpan, Dict[str, Any]]) -> Dict[str, Any]:
        """Enhance a span by adding context, fixing issues, or expanding content"""
        # Ensure initialization
        if not self.initialized:
            await self.initialize()
        
        # Convert to dict if needed
        if isinstance(span, DiamondSpan):
            span_dict = {
                "id": span.id,
                "parent_ids": span.parent_ids,
                "content": span.content,
                "metadata": span.metadata
            }
        else:
            span_dict = span
        
        # Prepare enhancement prompt
        prompt = f"""
        # DIAMOND SPAN ENHANCEMENT
        
        Please enhance the following Diamond Span to maximize its value and quality:
        
        ```json
        {json.dumps(span_dict, indent=2)}
        ```
        
        Perform the following enhancements:
        1. Expand and improve content while maintaining original meaning
        2. Add relevant metadata or context
        3. Fix any structural or semantic issues
        4. Increase the span's cognitive value
        
        Provide the enhanced span as JSON with the same structure.
        Only respond with the enhanced JSON.
        """
        
        # Generate enhancement
        enhanced_text, _ = await self.llm.generate_text(prompt)
        
        # Parse result
        try:
            enhanced_span = json.loads(enhanced_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the text
            import re
            json_match = re.search(r'({.*})', enhanced_text, re.DOTALL)
            if json_match:
                try:
                    enhanced_span = json.loads(json_match.group(1))
                except:
                    # Return original if enhancement failed
                    logger.warning(f"Failed to enhance span {span_dict.get('id', 'unknown')}")
                    return span_dict
            else:
                # Return original if enhancement failed
                logger.warning(f"Failed to enhance span {span_dict.get('id', 'unknown')}")
                return span_dict
        
        # Preserve the original ID
        if "id" in span_dict:
            enhanced_span["id"] = span_dict["id"]
        
        logger.info(f"Enhanced span {enhanced_span.get('id', 'unknown')}")
        
        return enhanced_span
    
    async def generate_from_prompt(self, prompt: str, kind: str = None, 
                                parent_ids: List[str] = None) -> Dict[str, Any]:
        """Generate a new span from a text prompt"""
        # Ensure initialization
        if not self.initialized:
            await self.initialize()
        
        # Set defaults
        parent_str = ", ".join(parent_ids or ["none"])
        kind = kind or "diamond"
        
        # Create generation prompt
        gen_prompt = f"""
        # DIAMOND SPAN GENERATION
        
        Generate a high-quality Diamond Span based on the following:
        
        User prompt: "{prompt}"
        Span kind: {kind}
        Parent IDs: {parent_str}
        
        Requirements:
        - The span should have all required fields
        - Content should be rich and valuable
        - Include appropriate metadata
        
        Output the span as a JSON object with this structure:
        {{
          "id": "<auto_generated>",
          "kind": "{kind}",
          "verb": "<ACTION_VERB>",
          "actor": "<actor_name>",
          "object": "<target_object>",
          "parent_ids": [{parent_str}],
          "payload": {{
            // Relevant content based on kind
          }},
          "metadata": {{
            // Relevant metadata
          }}
        }}
        
        Only respond with the JSON object.
        """
        
        # Generate the span
        generated_text, stats = await self.llm.generate_text(gen_prompt)
        
        # Parse result
        try:
            generated_span = json.loads(generated_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the text
            import re
            json_match = re.search(r'({.*})', generated_text, re.DOTALL)
            if json_match:
                try:
                    generated_span = json.loads(json_match.group(1))
                except:
                    # Create a basic span if parsing fails
                    generated_span = {
                        "id": f"{kind}-{int(datetime.now().timestamp())}",
                        "kind": kind,
                        "verb": "STATE",
                        "actor": "llm",
                        "object": "content",
                        "parent_ids": parent_ids or [],
                        "payload": {"text": prompt},
                        "metadata": {
                            "generated_by": "llm",
                            "generation_failed": True,
                            "timestamp": datetime.now().isoformat()
                        }
                    }
            else:
                # Create a basic span if parsing fails
                generated_span = {
                    "id": f"{kind}-{int(datetime.now().timestamp())}",
                    "kind": kind,
                    "verb": "STATE",
                    "actor": "llm",
                    "object": "content",
                    "parent_ids": parent_ids or [],
                    "payload": {"text": prompt},
                    "metadata": {
                        "generated_by": "llm",
                        "generation_failed": True,
                        "timestamp": datetime.now().isoformat()
                    }
                }
        
        # Add generation stats
        if "metadata" not in generated_span:
            generated_span["metadata"] = {}
        
        generated_span["metadata"]["generation_stats"] = {
            "tokens": stats.tokens_generated,
            "time_ms": stats.generation_time_ms,
            "tokens_per_second": stats.tokens_per_second,
            "model": stats.model_id,
            "timestamp": stats.timestamp
        }
        
        logger.info(f"Generated new span of kind {kind} from prompt")
        
        return generated_span
    
    async def _process_diamond_span(self, span_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process a diamond span"""
        # Add a quality score if not present
        if "metadata" not in span_dict:
            span_dict["metadata"] = {}
        
        if "quality_score" not in span_dict["metadata"]:
            validation = await self.validate_span(span_dict)
            span_dict["metadata"]["quality_score"] = validation.get("quality_score", 5.0)
        
        return span_dict
    
    async def _process_training_span(self, span_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process a training-related span"""
        # Add expected outcomes if not present
        if "payload" not in span_dict:
            span_dict["payload"] = {}
        
        if "expected_outcomes" not in span_dict["payload"]:
            # Generate expected outcomes based on training parameters
            payload = span_dict["payload"]
            
            prompt = f"""
            # TRAINING SPAN OUTCOME PREDICTION
            
            Based on the following training parameters, predict the expected outcomes:
            
            ```json
            {json.dumps(payload, indent=2)}
            ```
            
            Generate only a JSON object with the expected outcomes:
            {{
              "expected_accuracy": 0.0-1.0,
              "expected_loss": 0.0-10.0,
              "expected_duration": "time in hours",
              "expected_improvements": ["area1", "area2", ...],
              "potential_issues": ["issue1", "issue2", ...]
            }}
            """
            
            # Generate prediction
            prediction_text, _ = await self.llm.generate_text(prompt)
            
            # Parse result
            try:
                outcomes = json.loads(prediction_text)
                span_dict["payload"]["expected_outcomes"] = outcomes
            except:
                # Use default outcomes if parsing fails
                span_dict["payload"]["expected_outcomes"] = {
                    "expected_accuracy": 0.85,
                    "expected_loss": 2.3,
                    "expected_duration": "unknown",
                    "expected_improvements": ["general performance"],
                    "potential_issues": ["overfitting", "resource constraints"]
                }
        
        return span_dict
    
    async def _process_prediction_span(self, span_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process a prediction span"""
        # No special processing yet
        return span_dict
    
    async def _process_governance_span(self, span_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process a governance span"""
        # Add impact assessment if not present
        if "payload" not in span_dict:
            span_dict["payload"] = {}
        
        if "impact_assessment" not in span_dict["payload"]:
            # Generate impact assessment
            payload = span_dict["payload"]
            
            prompt = f"""
            # GOVERNANCE IMPACT ASSESSMENT
            
            Based on the following governance policy, assess its potential impacts:
            
            ```json
            {json.dumps(payload, indent=2)}
            ```
            
            Generate only a JSON object with the impact assessment:
            {{
              "impact_score": 1-10,
              "affected_areas": ["area1", "area2", ...],
              "benefits": ["benefit1", "benefit2", ...],
              "risks": ["risk1", "risk2", ...],
              "recommendations": ["rec1", "rec2", ...]
            }}
            """
            
            # Generate assessment
            assessment_text, _ = await self.llm.generate_text(prompt)
            
            # Parse result
            try:
                assessment = json.loads(assessment_text)
                span_dict["payload"]["impact_assessment"] = assessment
            except:
                # Use default assessment if parsing fails
                span_dict["payload"]["impact_assessment"] = {
                    "impact_score": 5,
                    "affected_areas": ["general governance"],
                    "benefits": ["improved structure"],
                    "risks": ["implementation challenges"],
                    "recommendations": ["careful monitoring"]
                }
        
        return span_dict
    
    async def _process_generic_span(self, span_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process a generic span"""
        # No special processing
        return span_dict