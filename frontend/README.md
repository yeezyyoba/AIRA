# AIRA Frontend

The AIRA frontend is a web application built with Next.js, React, TypeScript, and Tailwind CSS. It provides the user interface for interacting with the Amharic Intelligent Research Assistant.

## Requirements

- Node.js
- npm
- A running AIRA FastAPI backend

## Installation

From the `frontend/` directory, install dependencies:

```bash
npm ci
```

Create the local environment file:

```bash
cp .env.example .env.local
```

Configure the backend API URL:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Development

Start the development server:

```bash
npm run dev
```

Open:

http://localhost:3000

## Production Build

Build the frontend:

```bash
npm run build
```

Start the production server:

```bash
npm run start
```

## Linting

Run:

```bash
npm run lint
```

## Backend Integration

The frontend communicates with the AIRA FastAPI backend.

The backend should be running at:

http://localhost:8000

API documentation is available at:

http://localhost:8000/docs

## Technology Stack

- Next.js
- React
- TypeScript
- Tailwind CSS
- Lucide React
