// Tensor Engine implementation in LogLine programming language
// Created: 2025-07-19 05:54:11
// Author: danvoulez

// Import core LogLine libraries
import "core/base.ll"
import "core/memory.ll"
import "core/diamond.ll"
import "core/parallel.ll"

// TensorShape defines the shape of a tensor
type TensorShape {
    dimensions: u32
    sizes: []u32
    stride: []u32
    total_size: u32
}

// TensorDataType defines the data type of tensor elements
enum TensorDataType {
    FLOAT32 = 0
    FLOAT16 = 1
    INT32 = 2
    INT64 = 3
    BOOL = 4
    COMPLEX64 = 5
}

// Device where tensor operations will be executed
enum TensorDevice {
    CPU = 0
    CUDA = 1
    MPS = 2
    TPU = 3
}

// TensorDescriptor contains metadata about a tensor
type TensorDescriptor {
    id: string
    data_type: TensorDataType
    shape: TensorShape
    device: TensorDevice
    is_contiguous: bool
    created_at: timestamp
    creator: string
    energy: f32
    span_id: optional<string>
}

// The actual tensor data and operations
type Tensor {
    descriptor: TensorDescriptor
    data: *void  // Pointer to raw data
    
    // Operations
    fn add(other: *Tensor) -> *Tensor
    fn sub(other: *Tensor) -> *Tensor
    fn mul(other: *Tensor) -> *Tensor
    fn div(other: *Tensor) -> *Tensor
    fn matmul(other: *Tensor) -> *Tensor
    fn transpose() -> *Tensor
    fn reshape(new_shape: []u32) -> *Tensor
    fn slice(start: []u32, end: []u32) -> *Tensor
    fn to_device(device: TensorDevice) -> *Tensor
    fn to_type(data_type: TensorDataType) -> *Tensor
}

// TensorEngine manages tensor operations and provides high-level functions
// It also integrates with Diamond Spans
module TensorEngine {
    // Configuration
    struct Config {
        default_device: TensorDevice = TensorDevice::CPU
        default_type: TensorDataType = TensorDataType::FLOAT32
        cache_enabled: bool = true
        cache_size: u32 = 100
        energy_factor: f32 = 0.5
        create_spans: bool = true
        parallel_operations: u32 = 4
    }
    
    // Statistics tracking
    struct Stats {
        total_operations: u64 = 0
        total_memory: u64 = 0
        total_energy: f32 = 0.0
        cache_hits: u64 = 0
        cache_misses: u64 = 0
        spans_created: u64 = 0
        operation_counts: map<string, u64>
        
        fn record_operation(name: string, memory: u64, energy: f32) {
            self.total_operations += 1
            self.total_memory += memory
            self.total_energy += energy
            
            if self.operation_counts.contains(name) {
                self.operation_counts[name] += 1
            } else {
                self.operation_counts[name] = 1
            }
        }
    }
    
    // Private state
    private var config: Config
    private var stats: Stats
    private var tensors: map<string, *Tensor>
    private var cache: map<string, *Tensor>
    private var tensor_to_span: map<string, string>
    
    // Initialize the engine
    fn init(cfg: optional<Config> = none) -> void {
        self.config = cfg or Config{}
        self.stats = Stats{}
        self.tensors = map<string, *Tensor>{}
        self.cache = map<string, *Tensor>{}
        self.tensor_to_span = map<string, string>{}
        
        // Create genesis span for the tensor engine
        if self.config.create_spans {
            let span_id = diamond.create_span(
                kind: "tensor_engine",
                verb: "INITIALIZE",
                actor: "system",
                object: "tensor_engine",
                payload: {
                    "config": self.config
                }
            )
            
            // Track the initialization span
            self.tensor_to_span["tensor_engine_init"] = span_id
        }
    }
    
    // Create a new tensor from data
    fn create_tensor<T>(
        data: []T, 
        shape: optional<[]u32> = none, 
        data_type: optional<TensorDataType> = none,
        device: optional<TensorDevice> = none,
        name: optional<string> = none,
        creator: string = "system"
    ) -> Result<*Tensor, string> {
        // Determine shape if not provided
        let actual_shape = shape or [data.len() as u32]
        
        // Calculate total size
        let total_size = actual_shape.reduce(1, (acc, dim) => acc * dim)
        
        // Verify data size matches shape
        if data.len() != total_size as usize {
            return Err("Data size does not match shape")
        }
        
        // Determine data type
        let actual_type = data_type or match typeof(T) {
            f32 => TensorDataType::FLOAT32,
            f16 => TensorDataType::FLOAT16,
            i32 => TensorDataType::INT32,
            i64 => TensorDataType::INT64,
            bool => TensorDataType::BOOL,
            _ => self.config.default_type
        }
        
        // Determine device
        let actual_device = device or self.config.default_device
        
        // Generate ID
        let tensor_id = name or generate_id("tensor")
        
        // Create tensor descriptor
        let descriptor = TensorDescriptor{
            id: tensor_id,
            data_type: actual_type,
            shape: TensorShape{
                dimensions: actual_shape.len() as u32,
                sizes: actual_shape,
                stride: calculate_strides(actual_shape),
                total_size: total_size
            },
            device: actual_device,
            is_contiguous: true,
            created_at: now(),
            creator: creator,
            energy: calculate_base_energy(total_size, actual_type),
            span_id: none
        }
        
        // Allocate memory and copy data
        let element_size = data_type_size(actual_type)
        let memory_size = total_size * element_size
        let memory_ptr = memory.allocate(memory_size)
        
        // Copy data to allocated memory
        memory.copy(memory_ptr, data.ptr(), memory_size)
        
        // Create tensor object
        let tensor = new Tensor{
            descriptor: descriptor,
            data: memory_ptr
        }
        
        // Register tensor
        self.tensors[tensor_id] = tensor
        
        // Create a diamond span for this tensor if enabled
        if self.config.create_spans {
            let span_id = diamond.create_span(
                kind: "tensor",
                verb: "CREATE",
                actor: creator,
                object: tensor_id,
                payload: {
                    "shape": actual_shape,
                    "data_type": actual_type,
                    "device": actual_device,
                    "memory_size": memory_size
                }
            )
            
            // Record the span ID in the descriptor
            tensor.descriptor.span_id = span_id
            
            // Map tensor to span
            self.tensor_to_span[tensor_id] = span_id
            
            // Update statistics
            self.stats.spans_created += 1
        }
        
        // Update statistics
        self.stats.record_operation("create_tensor", memory_size, tensor.descriptor.energy)
        
        return Ok(tensor)
    }
    
    // Get a tensor by ID
    fn get_tensor(id: string) -> optional<*Tensor> {
        return self.tensors.get(id)
    }
    
    // Delete a tensor and free its memory
    fn delete_tensor(id: string, actor: string = "system") -> bool {
        let tensor = self.tensors.get(id)
        if tensor == none {
            return false
        }
        
        // Get tensor descriptor
        let descriptor = tensor.descriptor
        
        // Calculate memory size
        let element_size = data_type_size(descriptor.data_type)
        let memory_size = descriptor.shape.total_size * element_size
        
        // Free memory
        memory.free(tensor.data)
        
        // Create a diamond span for this operation if enabled
        if self.config.create_spans {
            let span_id = diamond.create_span(
                kind: "tensor",
                verb: "DELETE",
                actor: actor,
                object: id,
                payload: {
                    "memory_freed": memory_size,
                    "energy": descriptor.energy
                }
            )
            
            // If tensor had a creation span, link them
            if descriptor.span_id.has_value() {
                diamond.link_spans(descriptor.span_id.unwrap(), span_id)
            }
            
            // Update statistics
            self.stats.spans_created += 1
        }
        
        // Remove from registries
        self.tensors.remove(id)
        self.tensor_to_span.remove(id)
        
        // Update statistics (negative memory since we're freeing)
        self.stats.record_operation("delete_tensor", 0, descriptor.energy * 0.1)
        self.stats.total_memory -= memory_size
        
        return true
    }
    
    // Perform matrix multiplication
    fn matmul(
        a_id: string, 
        b_id: string, 
        output_id: optional<string> = none,
        actor: string = "system"
    ) -> Result<string, string> {
        // Get input tensors
        let tensor_a = self.tensors.get(a_id)
        let tensor_b = self.tensors.get(b_id)
        
        if tensor_a == none {
            return Err("Tensor A not found")
        }
        
        if tensor_b == none {
            return Err("Tensor B not found")
        }
        
        // Check shapes compatibility
        if tensor_a.descriptor.shape.dimensions < 2 or tensor_b.descriptor.shape.dimensions < 2 {
            return Err("Matmul requires at least 2D tensors")
        }
        
        let a_shape = tensor_a.descriptor.shape
        let b_shape = tensor_b.descriptor.shape
        
        // Check inner dimensions match
        if a_shape.sizes[a_shape.dimensions - 1] != b_shape.sizes[b_shape.dimensions - 2] {
            return Err("Incompatible tensor shapes for matmul")
        }
        
        // Start measuring time
        let start_time = now()
        
        // Try to use cache
        let cache_key = if self.config.cache_enabled {
            generate_cache_key("matmul", [a_id, b_id])
        } else {
            ""
        }
        
        let result_id = output_id or generate_id("tensor")
        
        // Check cache
        if self.config.cache_enabled and self.cache.contains(cache_key) {
            // Cache hit - copy the cached tensor
            let cached = self.cache[cache_key]
            let result = clone_tensor(cached, result_id)
            
            // Register result tensor
            self.tensors[result_id] = result
            
            // Update statistics
            self.stats.cache_hits += 1
            
            return Ok(result_id)
        }
        
        // Cache miss - perform operation
        self.stats.cache_misses += 1
        
        // Calculate output shape
        let output_shape = []u32
        
        // For matrices: result is (M x N) for A(M x K) * B(K x N)
        if a_shape.dimensions == 2 and b_shape.dimensions == 2 {
            output_shape = [a_shape.sizes[0], b_shape.sizes[1]]
        } else {
            // For batched matrices, need more complex shape calculation
            // (omitted for brevity)
            return Err("Batched matmul not implemented in this example")
        }
        
        // Calculate operation complexity and energy
        let m = a_shape.sizes[0]
        let k = a_shape.sizes[1]
        let n = b_shape.sizes[1]
        let flops = 2 * m * n * k  // Approx. number of floating-point operations
        let energy = calculate_operation_energy("matmul", flops)
        
        // Create output tensor descriptor
        let output_descriptor = TensorDescriptor{
            id: result_id,
            data_type: tensor_a.descriptor.data_type,  // Use same type as input
            shape: TensorShape{
                dimensions: 2,
                sizes: output_shape,
                stride: calculate_strides(output_shape),
                total_size: output_shape[0] * output_shape[1]
            },
            device: tensor_a.descriptor.device,  // Use same device as input
            is_contiguous: true,
            created_at: now(),
            creator: actor,
            energy: energy,
            span_id: none
        }
        
        // Allocate memory for result
        let element_size = data_type_size(output_descriptor.data_type)
        let memory_size = output_descriptor.shape.total_size * element_size
        let result_ptr = memory.allocate(memory_size)
        
        // Execute matrix multiplication based on device
        match output_descriptor.device {
            TensorDevice::CPU => {
                // Execute CPU matmul
                execute_cpu_matmul(
                    tensor_a.data, a_shape,
                    tensor_b.data, b_shape,
                    result_ptr, output_descriptor.shape
                )
            },
            TensorDevice::CUDA => {
                // Execute CUDA matmul if available
                execute_gpu_matmul(
                    tensor_a.data, a_shape,
                    tensor_b.data, b_shape,
                    result_ptr, output_descriptor.shape
                )
            },
            _ => {
                // Fallback to CPU
                execute_cpu_matmul(
                    tensor_a.data, a_shape,
                    tensor_b.data, b_shape,
                    result_ptr, output_descriptor.shape
                )
            }
        }
        
        // Create result tensor
        let result_tensor = new Tensor{
            descriptor: output_descriptor,
            data: result_ptr
        }
        
        // Register result tensor
        self.tensors[result_id] = result_tensor
        
        // Calculate execution time
        let execution_time = now() - start_time
        
        // Create diamond span for the operation
        if self.config.create_spans {
            let parent_ids = []string
            
            // Add parent spans if available
            if tensor_a.descriptor.span_id.has_value() {
                parent_ids.push(tensor_a.descriptor.span_id.unwrap())
            }
            
            if tensor_b.descriptor.span_id.has_value() {
                parent_ids.push(tensor_b.descriptor.span_id.unwrap())
            }
            
            let span_id = diamond.create_span(
                kind: "tensor_operation",
                verb: "MATMUL",
                actor: actor,
                object: result_id,
                parent_ids: parent_ids,
                payload: {
                    "operation": "matmul",
                    "input_tensors": [a_id, b_id],
                    "output_tensor": result_id,
                    "flops": flops,
                    "execution_time_ms": execution_time * 1000.0,
                    "energy": energy
                }
            )
            
            // Record span ID in descriptor and maps
            result_tensor.descriptor.span_id = span_id
            self.tensor_to_span[result_id] = span_id
            
            // Update statistics
            self.stats.spans_created += 1
        }
        
        // Update cache if enabled
        if self.config.cache_enabled {
            self.cache[cache_key] = result_tensor
            
            // Limit cache size
            if self.cache.size() > self.config.cache_size {
                let oldest = self.cache.keys()[0]  // Simplification - should find actual oldest
                self.cache.remove(oldest)
            }
        }
        
        // Update statistics
        self.stats.record_operation("matmul", memory_size, energy)
        
        return Ok(result_id)
    }
    
    // Apply element-wise operation
    fn elementwise_op(
        op: string,
        a_id: string,
        b_id: string,
        output_id: optional<string> = none,
        actor: string = "system"
    ) -> Result<string, string> {
        // Implementation similar to matmul but for element-wise operations
        // Abbreviated for brevity
        
        let result_id = output_id or generate_id("tensor")
        
        // (Implementation would check tensors, shapes, perform operation, etc.)
        
        // Update statistics
        self.stats.record_operation(op, 0, 0.0)
        
        return Ok(result_id)
    }
    
    // Add two tensors element-wise
    fn add(a_id: string, b_id: string, output_id: optional<string> = none, actor: string = "system") -> Result<string, string> {
        return self.elementwise_op("add", a_id, b_id, output_id, actor)
    }
    
    // Subtract two tensors element-wise
    fn sub(a_id: string, b_id: string, output_id: optional<string> = none, actor: string = "system") -> Result<string, string> {
        return self.elementwise_op("sub", a_id, b_id, output_id, actor)
    }
    
    // Multiply two tensors element-wise
    fn mul(a_id: string, b_id: string, output_id: optional<string> = none, actor: string = "system") -> Result<string, string> {
        return self.elementwise_op("mul", a_id, b_id, output_id, actor)
    }
    
    // Divide two tensors element-wise
    fn div(a_id: string, b_id: string, output_id: optional<string> = none, actor: string = "system") -> Result<string, string> {
        return self.elementwise_op("div", a_id, b_id, output_id, actor)
    }
    
    // Get statistics
    fn get_stats() -> Stats {
        return self.stats
    }
    
    // Get tensor info without data
    fn get_tensor_info(id: string) -> optional<TensorDescriptor> {
        let tensor = self.tensors.get(id)
        if tensor == none {
            return none
        }
        
        return tensor.descriptor
    }
    
    // Helper functions
    private fn generate_id(prefix: string) -> string {
        let timestamp = now() as u64
        let random_part = uuid().substring(0, 8)
        return prefix + "-" + timestamp as string + "-" + random_part
    }
    
    private fn generate_cache_key(op: string, tensor_ids: []string) -> string {
        let key = op
        for id in tensor_ids {
            key += ":" + id
        }
        return key
    }
    
    private fn calculate_strides(shape: []u32) -> []u32 {
        let dims = shape.len()
        let strides = []u32(dims)
        
        let current_stride = 1
        for i in (0..dims).reverse() {
            strides[i] = current_stride
            current_stride *= shape[i]
        }
        
        return strides
    }
    
    private fn data_type_size(dtype: TensorDataType) -> u32 {
        match dtype {
            TensorDataType::FLOAT32 => 4,
            TensorDataType::FLOAT16 => 2,
            TensorDataType::INT32 => 4,
            TensorDataType::INT64 => 8,
            TensorDataType::BOOL => 1,
            TensorDataType::COMPLEX64 => 8
        }
    }
    
    private fn calculate_base_energy(total_size: u32, dtype: TensorDataType) -> f32 {
        let base = 1.0
        let size_factor = total_size as f32 / 1000.0
        let type_factor = match dtype {
            TensorDataType::FLOAT16 => 0.5,
            TensorDataType::FLOAT32 => 1.0,
            TensorDataType::INT32 => 0.7,
            TensorDataType::INT64 => 1.2,
            TensorDataType::BOOL => 0.1,
            TensorDataType::COMPLEX64 => 1.5
        }
        
        return base * size_factor * type_factor * self.config.energy_factor
    }
    
    private fn calculate_operation_energy(op: string, flops: u32) -> f32 {
        let op_factor = match op {
            "matmul" => 2.0,
            "conv" => 3.0,
            "add" => 0.5,
            "sub" => 0.5,
            "mul" => 0.7,
            "div" => 1.0,
            "transpose" => 1.0,
            _ => 1.0
        }
        
        let base_energy = flops as f32 / 10000.0
        return base_energy * op_factor * self.config.energy_factor
    }
    
    // External function declarations for backend implementations
    extern fn execute_cpu_matmul(a: *void, a_shape: TensorShape, b: *void, b_shape: TensorShape, out: *void, out_shape: TensorShape) -> void
    extern fn execute_gpu_matmul(a: *void, a_shape: TensorShape, b: *void, b_shape: TensorShape, out: *void, out_shape: TensorShape) -> void
    extern fn clone_tensor(source: *Tensor, new_id: string) -> *Tensor
}

// Example of tensor integration with Lingua Mater
module LinguaTensor {
    // Import core Lingua Mater
    import "core/lingua_mater.ll"
    
    // Create a grammar vector from text
    fn text_to_vector(text: string, dimensions: u32 = 768, actor: string = "system") -> Result<string, string> {
        // Create grammar vector using Lingua Mater
        let grammar_vector = LinguaMater.create_grammar_vector(text)
        
        // Convert to dense vector of specified dimensions
        let dense_vec = grammar_vector.to_dense_vector(dimensions)
        
        // Create tensor from vector
        let result = TensorEngine.create_tensor(
            data: dense_vec,
            shape: [dimensions],
            name: "gramvec-" + uuid().substring(0, 8),
            creator: actor
        )
        
        if result.is_err() {
            return Err(result.unwrap_err())
        }
        
        let tensor_id = result.unwrap().descriptor.id
        
        // Create a diamond span to track this operation
        let span_id = diamond.create_span(
            kind: "grammar_vector",
            verb: "VECTORIZE",
            actor: actor,
            object: "text",
            payload: {
                "text_snippet": text.substring(0, 50) + (text.len() > 50 ? "..." : ""),
                "dimensions": dimensions,
                "tensor_id": tensor_id
            }
        )
        
        // Link tensor to span
        if result.unwrap().descriptor.span_id.has_value() {
            diamond.link_spans(result.unwrap().descriptor.span_id.unwrap(), span_id)
        } else {
            result.unwrap().descriptor.span_id = span_id
        }
        
        return Ok(tensor_id)
    }
    
    // Convert a vector tensor back to text
    fn vector_to_text(tensor_id: string, quality: f32 = 0.8, actor: string = "system") -> Result<string, string> {
        // Get tensor
        let tensor_opt = TensorEngine.get_tensor(tensor_id)
        if tensor_opt == none {
            return Err("Tensor not found")
        }
        
        let tensor = tensor_opt.unwrap()
        
        // Convert tensor data to vector format for Lingua Mater
        let vector_data = extract_vector_data(tensor)
        
        // Create grammar vector from dense vector
        let grammar_vector = LinguaMater.grammar_vector_from_dense(vector_data)
        
        // Generate text from grammar vector
        let text = grammar_vector.to_natural_language(quality)
        
        // Create a diamond span to track this operation
        let span_id = diamond.create_span(
            kind: "grammar_vector",
            verb: "TEXTUALIZE",
            actor: actor,
            object: tensor_id,
            payload: {
                "text_length": text.len(),
                "quality": quality,
                "tensor_id": tensor_id
            }
        )
        
        // Link to tensor's span if available
        if tensor.descriptor.span_id.has_value() {
            diamond.link_spans(tensor.descriptor.span_id.unwrap(), span_id)
        }
        
        return Ok(text)
    }
    
    // Grammar comparison operation
    fn grammar_similarity(tensor_a_id: string, tensor_b_id: string, actor: string = "system") -> Result<f32, string> {
        // Get tensors
        let tensor_a_opt = TensorEngine.get_tensor(tensor_a_id)
        let tensor_b_opt = TensorEngine.get_tensor(tensor_b_id)
        
        if tensor_a_opt == none or tensor_b_opt == none {
            return Err("One or both tensors not found")
        }
        
        let tensor_a = tensor_a_opt.unwrap()
        let tensor_b = tensor_b_opt.unwrap()
        
        // Extract vector data
        let vec_a = extract_vector_data(tensor_a)
        let vec_b = extract_vector_data(tensor_b)
        
        // Compute cosine similarity
        let similarity = LinguaMater.compute_similarity(vec_a, vec_b)
        
        // Create a diamond span for this computation
        let span_id = diamond.create_span(
            kind: "grammar_comparison",
            verb: "COMPARE",
            actor: actor,
            object: "vectors",
            payload: {
                "tensor_a": tensor_a_id,
                "tensor_b": tensor_b_id,
                "similarity": similarity
            }
        )
        
        // Link to both tensor spans
        if tensor_a.descriptor.span_id.has_value() {
            diamond.link_spans(tensor_a.descriptor.span_id.unwrap(), span_id)
        }
        
        if tensor_b.descriptor.span_id.has_value() {
            diamond.link_spans(tensor_b.descriptor.span_id.unwrap(), span_id)
        }
        
        return Ok(similarity)
    }
    
    // Helper function to extract vector data from tensor
    private fn extract_vector_data(tensor: *Tensor) -> []f32 {
        // Implementation depends on internal tensor representation
        // This is a simplified placeholder
        let shape = tensor.descriptor.shape
        let data_ptr = tensor.data as *f32
        let vector = []f32(shape.total_size)
        
        // Copy data
        for i in 0..shape.total_size {
            vector[i] = data_ptr[i]
        }
        
        return vector
    }
}