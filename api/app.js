import express from "express";
import morgan from "morgan";
import cors from "cors";
import "dotenv/config";

import authRouter from "./routes/authRouter.js";
import yachtsRouter from "./routes/yachtsRouter.js";
import eventsRouter from "./routes/eventsRouter.js";

const { APP_PORT } = process.env;

const app = express();

app.use(morgan("tiny"));
app.use(cors());
app.use(express.json());
app.use(express.static("public"));

app.use("/api/auth", authRouter);
app.use("/api/yachts", yachtsRouter);
app.use("/api/events", eventsRouter);

app.use((_, res) => {
  res.status(404).json({ message: "Route not found" });
});

app.use((err, _, res, __) => {
  const { status = 500, message = "Server error" } = err;
  res.status(status).json({ message });
});

app.listen(APP_PORT, () => {
  console.log(`Server is running. Use our API on port ${APP_PORT}`);
});
