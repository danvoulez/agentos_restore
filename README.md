# AgentOS Fusion

## Overview

AgentOS Fusion is a multi-component system designed for building and deploying AI-powered applications. It integrates a robust backend API, a dedicated user management service, and a frontend interface.

## Features

- **Backend**: FastAPI-based backend with MongoDB and Redis integration.
- **Frontend**: Vue.js-based frontend with Vite for development and production builds.
- **User Management**: Dedicated service for handling user profiles, roles, and authentication.
- **WebSocket Support**: Real-time updates using WebSocket connections.
- **Task Queue**: Asynchronous task processing with Celery and Redis.

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```bash
   cd AgentOS_Fusion
   ```

3. Install dependencies for the backend:
   ```bash
   cd promptos_backend
   pip install -r requirements.txt
   ```

4. Install dependencies for the frontend:
   ```bash
   cd fusion_app
   yarn install
   ```

5. Start the development servers:
   - Backend: `uvicorn app.main:app --reload`
   - Frontend: `yarn dev`

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
