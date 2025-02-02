# Step 1: Use Node.js image to build the application
FROM node:16-alpine AS build

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy package.json and package-lock.json
COPY package.json package-lock.json ./

# Step 4: Install dependencies
RUN npm install

# Step 5: Copy the rest of the application code
COPY . .

# Step 6: Build the React application
RUN npm run build

# Step 7: Use Nginx to serve the static files
FROM nginx:alpine

# Step 8: Copy the build output to the Nginx HTML folder
COPY --from=build /app/build /usr/share/nginx/html

# Step 9: Expose the port Nginx will use to serve the app
EXPOSE 80

# Step 10: Start the Nginx server
CMD ["nginx", "-g", "daemon off;"]
