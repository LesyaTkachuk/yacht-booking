import express from "express";
import controllerWrapper from "../helpers/controllerWrapper.js";
import validateBody from "../helpers/validateBody.js";
import {
  getAllEvents,
  getEventById,
  createEvent,
  updateEventById,
  deleteEventById,
} from "../controllers/eventsController.js";
import {
  createEventSchema,
  updateEventSchema,
} from "../schemas/eventsSchema.js";
import authenticate from "../middlewares/authenticate.js";

const eventsRouter = express.Router();

eventsRouter.use(authenticate);

eventsRouter.get("/", controllerWrapper(getAllEvents));

eventsRouter.get("/:id", controllerWrapper(getEventById));

eventsRouter.post(
  "/",
  validateBody(createEventSchema),
  controllerWrapper(createEvent)
);

eventsRouter.put(
  "/:id",
  validateBody(updateEventSchema),
  controllerWrapper(updateEventById)
);

eventsRouter.delete("/:id", controllerWrapper(deleteEventById));

export default eventsRouter;
