import Joi from "joi";
import { EVENTS } from "../constants/events.js";

export const createEventSchema = Joi.object({
  type: Joi.string()
    .valid(
      EVENTS.VIEW,
      EVENTS.WISHLIST,
      EVENTS.CHAT_OWNER,
      EVENTS.START_BOOKING,
      EVENTS.BOOK
    )
    .required(),
  weight: Joi.number().optional(),
  yachtId: Joi.string().required(),
});

export const updateEventSchema = Joi.object({
  type: Joi.string().valid(
    EVENTS.VIEW,
    EVENTS.WISHLIST,
    EVENTS.CHAT_OWNER,
    EVENTS.START_BOOKING,
    EVENTS.BOOK
  ),
  weight: Joi.number(),
  yachtId: Joi.string(),
})
  .min(1)
  .message("Body must have at least one field");
