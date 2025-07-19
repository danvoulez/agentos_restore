"""
Enhanced Input Validation for LogLineOS
Provides comprehensive input validation with sanitization
Created: 2025-07-19 06:00:41 UTC
User: danvoulez
"""
import re
import json
import html
import base64
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable, Type, TypeVar
from pydantic import BaseModel, ValidationError, validator
import logging

# Configure logging
logger = logging.getLogger("InputValidation")

# Generic type for validator
T = TypeVar('T')

class ValidationResult:
    """Result of a validation check"""
    
    def __init__(self, valid: bool = True, errors: List[str] = None, sanitized_value: Any = None):
        self.valid = valid
        self.errors = errors or []
        self.sanitized_value = sanitized_value
    
    def __bool__(self):
        return self.valid


class InputValidator:
    """Advanced input validator with sanitization capabilities"""
    
    @staticmethod
    def validate_string(value: Any, 
                       min_length: int = 0, 
                       max_length: int = None, 
                       pattern: str = None,
                       sanitize: bool = True) -> ValidationResult:
        """
        Validate and sanitize string input
        
        Args:
            value: Input value
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            pattern: Regex pattern to match
            sanitize: Whether to sanitize the input
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        
        # Type check
        if not isinstance(value, str):
            return ValidationResult(
                valid=False, 
                errors=["Value must be a string"], 
                sanitized_value=str(value) if sanitize else value
            )
        
        # Sanitize if requested
        sanitized_value = html.escape(value) if sanitize else value
        
        # Length check
        if len(sanitized_value) < min_length:
            errors.append(f"String length must be at least {min_length} characters")
        
        if max_length is not None and len(sanitized_value) > max_length:
            errors.append(f"String length must not exceed {max_length} characters")
        
        # Pattern check
        if pattern and not re.match(pattern, sanitized_value):
            errors.append(f"String does not match required pattern")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            sanitized_value=sanitized_value
        )
    
    @staticmethod
    def validate_email(value: str) -> ValidationResult:
        """
        Validate email address
        
        Args:
            value: Email address to validate
            
        Returns:
            ValidationResult with validation details
        """
        # Email regex pattern
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        return InputValidator.validate_string(
            value,
            min_length=5,
            max_length=255,
            pattern=pattern,
            sanitize=True
        )
    
    @staticmethod
    def validate_numeric(value: Any, 
                        min_value: Union[int, float] = None, 
                        max_value: Union[int, float] = None,
                        integer_only: bool = False) -> ValidationResult:
        """
        Validate numeric input
        
        Args:
            value: Input value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            integer_only: Whether only integers are allowed
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        
        # Type conversion
        try:
            if integer_only:
                if not isinstance(value, int) and not (isinstance(value, str) and value.isdigit()):
                    errors.append("Value must be an integer")
                    return ValidationResult(valid=False, errors=errors, sanitized_value=value)
                
                sanitized_value = int(value)
            else:
                sanitized_value = float(value)
        except (ValueError, TypeError):
            return ValidationResult(
                valid=False,
                errors=["Value must be a number"],
                sanitized_value=value
            )
        
        # Range check
        if min_value is not None and sanitized_value < min_value:
            errors.append(f"Value must be at least {min_value}")
        
        if max_value is not None and sanitized_value > max_value:
            errors.append(f"Value must not exceed {max_value}")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            sanitized_value=sanitized_value
        )
    
    @staticmethod
    def validate_boolean(value: Any) -> ValidationResult:
        """
        Validate boolean input
        
        Args:
            value: Input value
            
        Returns:
            ValidationResult with validation details
        """
        # Direct boolean
        if isinstance(value, bool):
            return ValidationResult(valid=True, sanitized_value=value)
        
        # String conversions
        if isinstance(value, str):
            value_lower = value.lower()
            if value_lower in ("true", "1", "yes", "y", "on"):
                return ValidationResult(valid=True, sanitized_value=True)
            elif value_lower in ("false", "0", "no", "n", "off"):
                return ValidationResult(valid=True, sanitized_value=False)
        
        # Integer conversions
        if isinstance(value, int):
            if value == 1:
                return ValidationResult(valid=True, sanitized_value=True)
            elif value == 0:
                return ValidationResult(valid=True, sanitized_value=False)
        
        return ValidationResult(
            valid=False,
            errors=["Value must be a boolean"],
            sanitized_value=bool(value)
        )
    
    @staticmethod
    def validate_json(value: Any) -> ValidationResult:
        """
        Validate and sanitize JSON input
        
        Args:
            value: Input value (string or dict)
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        
        # Already a dict
        if isinstance(value, dict):
            return ValidationResult(valid=True, sanitized_value=value)
        
        # Parse JSON string
        if isinstance(value, str):
            try:
                sanitized_value = json.loads(value)
                return ValidationResult(valid=True, sanitized_value=sanitized_value)
            except json.JSONDecodeError:
                errors.append("Invalid JSON format")
        else:
            errors.append("Value must be a JSON string or dictionary")
        
        return ValidationResult(
            valid=False,
            errors=errors,
            sanitized_value=value
        )
    
    @staticmethod
    def validate_list(value: Any, 
                     item_validator: Callable[[Any], ValidationResult] = None,
                     min_length: int = 0,
                     max_length: int = None) -> ValidationResult:
        """
        Validate list input
        
        Args:
            value: Input value
            item_validator: Optional validator function for each item
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        
        # Type check
        if not isinstance(value, list):
            if isinstance(value, str):
                try:
                    # Try to parse JSON array
                    parsed_value = json.loads(value)
                    if isinstance(parsed_value, list):
                        value = parsed_value
                    else:
                        return ValidationResult(
                            valid=False,
                            errors=["Value must be a list"],
                            sanitized_value=[]
                        )
                except json.JSONDecodeError:
                    # Comma-separated list?
                    if ',' in value:
                        value = [item.strip() for item in value.split(',')]
                    else:
                        return ValidationResult(
                            valid=False,
                            errors=["Value must be a list"],
                            sanitized_value=[]
                        )
            else:
                return ValidationResult(
                    valid=False,
                    errors=["Value must be a list"],
                    sanitized_value=[]
                )
        
        # Length check
        if len(value) < min_length:
            errors.append(f"List must contain at least {min_length} items")
        
        if max_length is not None and len(value) > max_length:
            errors.append(f"List must not exceed {max_length} items")
        
        # Validate each item if validator provided
        sanitized_items = []
        if item_validator:
            item_errors = []
            
            for i, item in enumerate(value):
                item_result = item_validator(item)
                sanitized_items.append(item_result.sanitized_value)
                
                if not item_result.valid:
                    for error in item_result.errors:
                        item_errors.append(f"Item {i}: {error}")
            
            if item_errors:
                errors.extend(item_errors)
        else:
            sanitized_items = value
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            sanitized_value=sanitized_items
        )
    
    @staticmethod
    def validate_model(value: Any, model_class: Type[BaseModel]) -> ValidationResult:
        """
        Validate input against a Pydantic model
        
        Args:
            value: Input value (dict or JSON string)
            model_class: Pydantic model class
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        
        # Convert to dict if it's a string
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return ValidationResult(
                    valid=False,
                    errors=["Invalid JSON format"],
                    sanitized_value=value
                )
        
        # Validate against model
        try:
            model_instance = model_class(**value)
            return ValidationResult(
                valid=True,
                sanitized_value=model_instance.dict()
            )
        except ValidationError as e:
            for error in e.errors():
                location = ".".join(str(loc) for loc in error["loc"])
                errors.append(f"{location}: {error['msg']}")
        
        return ValidationResult(
            valid=False,
            errors=errors,
            sanitized_value=value
        )
    
    @staticmethod
    def sanitize_html(value: str) -> str:
        """
        Sanitize HTML content
        
        Args:
            value: HTML string to sanitize
            
        Returns:
            Sanitized HTML string
        """
        return html.escape(value)
    
    @staticmethod
    def validate_uuid(value: str) -> ValidationResult:
        """
        Validate UUID format
        
        Args:
            value: UUID string to validate
            
        Returns:
            ValidationResult with validation details
        