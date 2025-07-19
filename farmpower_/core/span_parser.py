"""
Span Parser for LogLineOS/DiamondSpan
Parses spans from text, files and binary formats
"""
import json
import os
import struct
import re
from typing import Dict, List, Any, Optional, Union, BinaryIO, TextIO

class SpanParser:
    """Parser for Diamond Spans"""
    
    @classmethod
    def from_text(cls, text: str) -> Dict[str, Any]:
        """Parse a span from text representation"""
        # Try JSON parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try YAML-like parsing
        try:
            return cls._parse_yaml_like(text)
        except Exception:
            pass
        
        # Try logline format parsing
        try:
            return cls._parse_logline_format(text)
        except Exception:
            pass
        
        # Last resort: treat as plain text
        return {
            "kind": "text",
            "verb": "STATE",
            "actor": "parser",
            "object": "content",
            "payload": {
                "text": text
            }
        }
    
    @classmethod
    def from_file(cls, file_path: str) -> Dict[str, Any]:
        """Parse a span from a file"""
        with open(file_path, 'rb') as f:
            # Check first few bytes to determine format
            header = f.read(4)
            f.seek(0)
            
            if header == b"DSPN":
                # It's a compiled span
                return cls._parse_binary_span(f)
            else:
                # Try as text
                try:
                    text = f.read().decode('utf-8')
                    return cls.from_text(text)
                except UnicodeDecodeError:
                    # It's binary but not our format
                    raise ValueError(f"Unsupported binary format in file: {file_path}")
    
    @classmethod
    def from_binary(cls, binary_data: bytes) -> Dict[str, Any]:
        """Parse a span from binary data"""
        import io
        with io.BytesIO(binary_data) as f:
            return cls._parse_binary_span(f)
    
    @classmethod
    def _parse_binary_span(cls, file_obj: BinaryIO) -> Dict[str, Any]:
        """Parse a binary span file"""
        # Read format identifier
        format_id = file_obj.read(4)
        if format_id != b"DSPN":
            raise ValueError(f"Invalid span format identifier: {format_id}")
        
        # Read version
        version = struct.unpack("!H", file_obj.read(2))[0]
        
        # Read metadata
        metadata_size = struct.unpack("!I", file_obj.read(4))[0]
        metadata_json = file_obj.read(metadata_size)
        metadata = json.loads(metadata_json)
        
        # Read binary data
        binary_size = struct.unpack("!I", file_obj.read(4))[0]
        binary_data = file_obj.read(binary_size)
        
        # Parse the binary data based on header
        if version == 1:
            header_size = struct.unpack("!I", binary_data[:4])[0]
            header_json = binary_data[4:4+header_size]
            header = json.loads(header_json)
            
            span_size = struct.unpack("!I", binary_data[4+header_size:8+header_size])[0]
            span_data = binary_data[8+header_size:8+header_size+span_size]
            
            if header.get("arch") == "apple_silicon":
                return cls._parse_apple_silicon_format(span_data, header, metadata)
            else:
                # Default format is just JSON
                return json.loads(span_data)
        else:
            raise ValueError(f"Unsupported binary format version: {version}")
    
    @classmethod
    def _parse_apple_silicon_format(cls, binary_data: bytes, 
                                  header: Dict[str, Any], 
                                  metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Apple Silicon optimized format"""
        # Check magic number
        if binary_data[:4] != b"ASPN":
            raise ValueError("Invalid Apple Silicon span format")
        
        # Read version
        version = struct.unpack("!H", binary_data[4:6])[0]
        
        # Start parsing fields
        pos = 6
        
        # Read span ID
        id_len = struct.unpack("!H", binary_data[pos:pos+2])[0]
        pos += 2
        span_id = binary_data[pos:pos+id_len].decode('utf-8')
        pos += id_len
        
        # Read kind, verb, actor, object
        fields = {}
        for field in ["kind", "verb", "actor", "object"]:
            field_len = struct.unpack("!H", binary_data[pos:pos+2])[0]
            pos += 2
            fields[field] = binary_data[pos:pos+field_len].decode('utf-8')
            pos += field_len
        
        # Read parent IDs
        parent_count = struct.unpack("!H", binary_data[pos:pos+2])[0]
        pos += 2
        parent_ids = []
        for _ in range(parent_count):
            pid_len = struct.unpack("!H", binary_data[pos:pos+2])[0]
            pos += 2
            parent_id = binary_data[pos:pos+pid_len].decode('utf-8')
            pos += pid_len
            parent_ids.append(parent_id)
        
        # Read payload
        payload_len = struct.unpack("!I", binary_data[pos:pos+4])[0]
        pos += 4
        payload_json = binary_data[pos:pos+payload_len].decode('utf-8')
        payload = json.loads(payload_json)
        pos += payload_len
        
        # Construct the span
        span = {
            "id": span_id,
            "kind": fields["kind"],
            "verb": fields["verb"],
            "actor": fields["actor"],
            "object": fields["object"],
            "parent_ids": parent_ids,
            "payload": payload,
            "metadata": metadata
        }
        
        return span
    
    @classmethod
    def _parse_yaml_like(cls, text: str) -> Dict[str, Any]:
        """Parse YAML-like format"""
        lines = text.strip().split('\n')
        result = {}
        current_key = None
        current_list = None
        current_dict = None
        
        for line in lines:
            line = line.rstrip()
            if not line or line.startswith('#'):
                continue
                
            if not line.startswith(' ') and ':' in line:
                # Top-level key
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if value:
                    result[key] = value
                else:
                    current_key = key
                    if key.endswith('_ids') or key.endswith('s'):
                        current_list = []
                        result[key] = current_list
                        current_dict = None
                    else:
                        current_dict = {}
                        result[key] = current_dict
                        current_list = None
            
            elif line.startswith(' ') and current_key:
                if current_list is not None:
                    # List item
                    if line.strip().startswith('-'):
                        item = line.strip()[1:].strip()
                        current_list.append(item)
                
                elif current_dict is not None:
                    # Dict item
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        current_dict[key.strip()] = value.strip()
        
        return result
    
    @classmethod
    def _parse_logline_format(cls, text: str) -> Dict[str, Any]:
        """Parse logline format"""
        lines = text.strip().split('\n')
        result = {"payload": {}}
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if line.startswith('logline:'):
                result["kind"] = line.split(':', 1)[1].strip()
            elif line.startswith('kind:'):
                result["kind"] = line.split(':', 1)[1].strip()
            elif line.startswith('who:'):
                result["actor"] = line.split(':', 1)[1].strip().strip('"')
            elif line.startswith('what:'):
                result["verb"] = line.split(':', 1)[1].strip().strip('"')
            elif line.startswith('why:'):
                result["payload"]["reason"] = line.split(':', 1)[1].strip().strip('"')
            elif line.startswith('where:'):
                result["object"] = line.split(':', 1)[1].strip().strip('"')
            elif line.startswith('payload:'):
                # The rest is payload
                payload_idx = lines.index(line)
                payload_text = '\n'.join(lines[payload_idx+1:])
                try:
                    result["payload"] = cls._parse_yaml_like(payload_text)
                except Exception:
                    result["payload"]["raw"] = payload_text
        
        return result