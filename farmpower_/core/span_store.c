/**
 * span_store.c - High-performance Diamond Span storage engine
 * 
 * This file implements the core storage mechanism for Diamond Spans,
 * optimized for fast access, durability, and atomic operations.
 * 
 * LogLineOS / Diamond Span Farm Project
 * Copyright (c) 2025 danvoulez
 * 
 * Last updated: 2025-07-19
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>