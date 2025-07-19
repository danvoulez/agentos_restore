"""
Checkpoint Manager for LogLineOS/DiamondSpan
Handles saving and loading model checkpoints with version control
"""
import os
import json
import logging
import asyncio
import hashlib
import shutil
import zipfile
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import torch

# Configure logging
logger = logging.getLogger("CheckpointManager")

class CheckpointManager:
    """
    Manager for LLM checkpoints with versioning and storage optimization
    """
    
    def __init__(self, base_path: str = "checkpoints"):
        self.base_path = base_path
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        self.metadata_path = os.path.join(base_path, "metadata.json")
        
        # Ensure base path exists
        os.makedirs(base_path, exist_ok=True)
        
        # Load metadata if exists
        self._load_metadata()
    
    def _load_metadata(self):
        """Load checkpoint metadata"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    self.checkpoints = json.load(f)
                logger.info(f"Loaded metadata for {len(self.checkpoints)} checkpoints")
            except Exception as e:
                logger.error(f"Error loading checkpoint metadata: {str(e)}")
                self.checkpoints = {}
    
    def _save_metadata(self):
        """Save checkpoint metadata"""
        with open(self.metadata_path, "w") as f:
            json.dump(self.checkpoints, f, indent=2)
    
    async def save_checkpoint(self, model, tokenizer, name: str, 
                            description: str = None) -> Dict[str, Any]:
        """Save a model checkpoint"""
        # Generate checkpoint version and path
        timestamp = int(time.time())
        version = f"v{timestamp}"
        checkpoint_id = f"{name}-{version}"
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.base_path, checkpoint_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save the model and tokenizer
        try:
            logger.info(f"Saving checkpoint {checkpoint_id}...")
            
            # Save model weights
            model_path = os.path.join(checkpoint_dir, "model")
            os.makedirs(model_path, exist_ok=True)
            
            # Handle different model types
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
            else:
                torch.save(model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))
            
            # Calculate checkpoint size
            checkpoint_size = self._get_directory_size(checkpoint_dir)
            
            # Create checkpoint metadata
            checkpoint_meta = {
                "id": checkpoint_id,
                "name": name,
                "version": version,
                "timestamp": timestamp,
                "created_at": datetime.now().isoformat(),
                "description": description or f"Checkpoint for {name}",
                "size_bytes": checkpoint_size,
                "path": checkpoint_dir
            }
            
            # Save to checkpoints registry
            self.checkpoints[checkpoint_id] = checkpoint_meta
            self._save_metadata()
            
            logger.info(f"Checkpoint {checkpoint_id} saved ({checkpoint_size / 1024 / 1024:.2f} MB)")
            
            return checkpoint_meta
            
        except Exception as e:
            logger.error(f"Error saving checkpoint {checkpoint_id}: {str(e)}", exc_info=True)
            # Clean up on failure
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            raise
    
    async def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load a model checkpoint"""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        checkpoint_meta = self.checkpoints[checkpoint_id]
        checkpoint_dir = checkpoint_meta["path"]
        
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        logger.info(f"Loading checkpoint {checkpoint_id}...")
        
        # The actual model loading would be done by the caller
        # This method just returns the metadata and verifies the checkpoint exists
        
        return checkpoint_meta
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint"""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        checkpoint_meta = self.checkpoints[checkpoint_id]
        checkpoint_dir = checkpoint_meta["path"]
        
        try:
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            
            # Remove from registry
            del self.checkpoints[checkpoint_id]
            self._save_metadata()
            
            logger.info(f"Checkpoint {checkpoint_id} deleted")
            return True
        except Exception as e:
            logger.error(f"Error deleting checkpoint {checkpoint_id}: {str(e)}")
            return False
    
    async def list_checkpoints(self, filter_name: str = None) -> List[Dict[str, Any]]:
        """List available checkpoints"""
        checkpoints = list(self.checkpoints.values())
        
        # Apply name filter if provided
        if filter_name:
            checkpoints = [cp for cp in checkpoints if filter_name in cp["name"]]
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda cp: cp["timestamp"], reverse=True)
        
        return checkpoints
    
    async def get_latest_checkpoint(self, name: str) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint for a given name"""
        checkpoints = await self.list_checkpoints(filter_name=name)
        return checkpoints[0] if checkpoints else None
    
    async def create_archive(self, checkpoint_id: str, output_path: str) -> str:
        """Create a portable archive of a checkpoint"""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        checkpoint_meta = self.checkpoints[checkpoint_id]
        checkpoint_dir = checkpoint_meta["path"]
        
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Create archive filename
        archive_filename = f"{checkpoint_id}.zip"
        archive_path = os.path.join(output_path, archive_filename)
        
        # Create the archive
        try:
            logger.info(f"Creating archive for checkpoint {checkpoint_id}...")
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(checkpoint_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(
                            file_path, 
                            os.path.relpath(file_path, os.path.dirname(checkpoint_dir))
                        )
            
            logger.info(f"Archive created: {archive_path}")
            return archive_path
            
        except Exception as e:
            logger.error(f"Error creating archive for checkpoint {checkpoint_id}: {str(e)}")
            if os.path.exists(archive_path):
                os.remove(archive_path)
            raise
    
    async def import_from_archive(self, archive_path: str) -> Dict[str, Any]:
        """Import a checkpoint from an archive"""
        if not os.path.exists(archive_path):
            raise ValueError(f"Archive not found: {archive_path}")
        
        # Extract checkpoint ID from filename
        checkpoint_id = os.path.splitext(os.path.basename(archive_path))[0]
        
        # Create extraction directory
        extract_dir = os.path.join(self.base_path, checkpoint_id)
        
        try:
            logger.info(f"Importing checkpoint from archive: {archive_path}...")
            
            # Extract archive
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(self.base_path)
            
            # Check for metadata.json in the extracted directory
            metadata_path = os.path.join(extract_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    checkpoint_meta = json.load(f)
            else:
                # Create basic metadata
                checkpoint_meta = {
                    "id": checkpoint_id,
                    "name": checkpoint_id.split("-v")[0],
                    "version": checkpoint_id.split("-")[-1],
                    "timestamp": int(time.time()),
                    "created_at": datetime.now().isoformat(),
                    "description": f"Imported checkpoint {checkpoint_id}",
                    "size_bytes": self._get_directory_size(extract_dir),
                    "path": extract_dir,
                    "imported": True
                }
            
            # Add to registry
            self.checkpoints[checkpoint_id] = checkpoint_meta
            self._save_metadata()
            
            logger.info(f"Checkpoint {checkpoint_id} imported")
            return checkpoint_meta
            
        except Exception as e:
            logger.error(f"Error importing checkpoint from archive {archive_path}: {str(e)}")
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            raise
    
    def _get_directory_size(self, path: str) -> int:
        """Calculate total size of a directory in bytes"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size