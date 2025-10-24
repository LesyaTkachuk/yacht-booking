import Event from "../db/models/Event.js";

export const createEvent = async (data) => {
  const event = await Event.create(data);
  return event;
};

export const listEvents = async (query) => {
  const events = await Event.findAll({ where: query });
  return events;
};

export const removeEvent = async (query) => {
  const event = await Event.destroy({ where: query });
  return event;
};

export const updateEvent = async (query, data) => {
  const event = await Event.update(data, { where: query });
  return event;
};

export const getEventById = async (id) => {
  const event = await Event.findOne({ where: { id } });
  return event;
};
