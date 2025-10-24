import * as eventsService from "../services/eventsService.js";
import { ACCESS_DENIED, EVENT_NOT_FOUND } from "../constants/errorMessages.js";
import { EVENT_DELETED } from "../constants/successMessages.js";
import { USER_ROLES } from "../constants/auth.js";
import { EVENT_WEIGHTS } from "../constants/events.js";

import HttpError from "../helpers/HttpError.js";

export const createEvent = async (req, res) => {
  const userId = req.user.id;

  let weight = req.body.weight;

  if (!weight) {
    weight = EVENT_WEIGHTS[req.body.type];
  }

  const event = await eventsService.createEvent({
    userId,
    ...req.body,
    weight,
  });
  res.status(201).json(event);
};

// TODO add filters and pagination
export const getAllEvents = async (req, res) => {
  const userRole = req.user.role;

  if (userRole !== USER_ROLES.ADMIN) {
    throw HttpError(403, ACCESS_DENIED);
  }

  const queryParams = req.query;

  const events = await eventsService.listEvents({
    ...(queryParams && queryParams),
  });
  res.json(events);
};

export const getEventById = async (req, res) => {
  const userRole = req.user.role;

  if (userRole !== USER_ROLES.ADMIN) {
    throw HttpError(403, ACCESS_DENIED);
  }
  const { id } = req.params;
  const event = await eventsService.getEvent({ id });
  if (!event) {
    throw HttpError(404, EVENT_NOT_FOUND);
  }
  res.json(event);
};

export const deleteEventById = async (req, res) => {
  const userRole = req.user.role;

  if (userRole !== USER_ROLES.ADMIN) {
    throw HttpError(403, ACCESS_DENIED);
  }
  const { id } = req.params;
  const event = await eventsService.removeEvent({ id });
  if (!event) {
    throw HttpError(404, EVENT_NOT_FOUND);
  }
  res.json({ message: EVENT_DELETED });
};

export const updateEventById = async (req, res) => {
  const userRole = req.user.role;

  if (userRole !== USER_ROLES.ADMIN) {
    throw HttpError(403, ACCESS_DENIED);
  }
  const { id } = req.params;
  const event = await eventsService.updateEvent({ id }, req.body);
  if (!event) {
    throw HttpError(404, EVENT_NOT_FOUND);
  }
  res.json(event);
};
