# My React App

This is a React application that includes a Dockerfile for easy deployment. You can run the app locally in a development environment or containerize it using Docker.

## Prerequisites

Before you begin, make sure you have the following installed:

- [Node.js](https://nodejs.org/en/download/) (version 16 or higher)
- [npm](https://www.npmjs.com/get-npm) (usually comes with Node.js)
- [Docker](https://www.docker.com/get-started)

## Running the App Locally

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/RMWeerasinghe/FYP_ARSynopsis.git
cd repo-name/client-side-application
```

### 2. Installing Dependencies
```bash
npm install
```
### 3. Starting the Development Server
```bash
npm start
```

The app will now be running at http://localhost:3000

## Running the App with Docker
If you'd prefer to run the app in a Docker container, follow these steps.Before these steps make sure to run the Docker Dekstop

## 1. Build Docker Image
run the following command to build the Docker image:
```bash
docker build -t my-react-app .
```

## 2. Run the Docker Container
Once the image is built, you can run it using the following command:
```bash
docker run -d -p 8080:80 --name react-container my-react-app
```

## 3. Access the App
Once the container is running, open your browser and go to http://localhost:8080 to see the app in action.