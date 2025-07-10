# TrainView

**TrainView** is a lightweight tool for visualizing neural network training metrics in real time. It combines a custom NumPy-based MLP implementation with a React frontend to display live-updating loss and accuracy curves during training.

---

## Overview

- Built a minimal MLP (1 hidden layer) using NumPy
- Designed a threaded FastAPI backend to handle training with interrupt/reset capability
- Developed a frontend in React with live polling and dynamic charts (via Recharts)
- Supports user-specified hyperparameters: optimizer, learning rate, batch size, epochs

---

## Features

- Live training visualization with chart updates per epoch
- Model restart on repeated training runs
- Clear UI for adjusting basic hyperparameters

---

## Usage

# Backend
python -m uvicorn backend.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
