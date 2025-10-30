import express from "express";
import {
  getAllYachts,
  getAllOwnYachts,
  getYachtById,
  deleteYachtById,
  createYacht,
  updateYachtById,
  updateYachtRatingById,
  getRecommendations,
} from "../controllers/yachtsController.js";
import controllerWrapper from "../helpers/controllerWrapper.js";
import validateBody from "../helpers/validateBody.js";
import {
  createYachtSchema,
  updateYachtSchema,
  updateYachtRatingSchema,
} from "../schemas/yachtsSchemas.js";
import authenticate from "../middlewares/authenticate.js";

const yachtsRouter = express.Router();

// TODO change all endpoints
yachtsRouter.get("/", controllerWrapper(getAllYachts));

yachtsRouter.get(
  "/recommendations",
  authenticate,
  controllerWrapper(getRecommendations)
);

yachtsRouter.get("/own", authenticate, controllerWrapper(getAllOwnYachts));

yachtsRouter.get("/:id", controllerWrapper(getYachtById));

yachtsRouter.delete("/:id", authenticate, controllerWrapper(deleteYachtById));

yachtsRouter.post(
  "/",
  authenticate,
  validateBody(createYachtSchema),
  controllerWrapper(createYacht)
);

yachtsRouter.put(
  "/:id",
  authenticate,
  validateBody(updateYachtSchema),
  controllerWrapper(updateYachtById)
);

yachtsRouter.patch(
  "/:id/rating",
  authenticate,
  validateBody(updateYachtRatingSchema),
  controllerWrapper(updateYachtRatingById)
);

export default yachtsRouter;
