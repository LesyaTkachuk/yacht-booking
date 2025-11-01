import * as yachtsService from "../services/yachtsService.js";
import { getYachtNotFoundMessage } from "../constants/errorMessages.js";
import { YACHT_DELETED } from "../constants/successMessages.js";
import { PERMISSION_DENIED } from "../constants/errorMessages.js";
import { USER_ROLES } from "../constants/auth.js";

import HttpError from "../helpers/HttpError.js";

// TODO add filters and pagination
export const getAllYachts = async (req, res) => {
  const queryParams = req.query;

  const yachts = await yachtsService.listYachts({
    ...(queryParams && queryParams),
  });
  res.json(yachts);
};

// TODO add filters and pagination
export const getAllOwnYachts = async (req, res) => {
  const queryParams = req.query;

  const userId = req.user.id;
  const yachts = await yachtsService.listYachts({
    userId,
    ...(queryParams && queryParams),
  });
  res.json(yachts);
};

export const getYachtById = async (req, res) => {
  const { id } = req.params;
  const yacht = await yachtsService.getYacht({ id });

  if (!yacht) {
    throw HttpError(404, getYachtNotFoundMessage(id));
  }
  res.json(yacht);
};

export const deleteYachtById = async (req, res) => {
  const { id: userId, role } = req.user;

  if (role === USER_ROLES.LESSEE) {
    throw HttpError(403, PERMISSION_DENIED);
  }

  const { id } = req.params;
  const yacht = await yachtsService.removeYacht({
    id,
    ...(role === USER_ROLES.LESSER && { userId }),
  });
  if (!yacht) {
    throw HttpError(404, getYachtNotFoundMessage(id));
  }

  res.json({ message: YACHT_DELETED });
};

export const createYacht = async (req, res) => {
  const { id: userId, role } = req.user;

  if (role !== USER_ROLES.LESSER) {
    throw HttpError(403, PERMISSION_DENIED);
  }

  const yacht = await yachtsService.addYacht({ userId, ...req.body });
  res.status(201).json(yacht);
};

export const updateYachtById = async (req, res) => {
  const { id: userId, role } = req.user;

  if (role === USER_ROLES.LESSEE) {
    throw HttpError(403, PERMISSION_DENIED);
  }
  const { id } = req.params;

  const yacht = await yachtsService.updateYacht({ userId, id }, req.body);

  if (!yacht) {
    throw HttpError(404, getYachtNotFoundMessage(id));
  }
  res.json(yacht);
};

export const updateYachtRatingById = async (req, res) => {
  const { id, role } = req.params;

  if (role !== USER_ROLES.ADMIN) {
    throw HttpError(403, PERMISSION_DENIED);
  }

  const rating = req.body.rating;
  const yacht = await yachtsService.updateYachtRating({ id }, rating);
  if (!yacht) {
    throw HttpError(404, getYachtNotFoundMessage(id));
  }
  res.json(contact);
};

export const getRecommendations = async (req, res) => {
  const { id } = req.user;
  const recommendations = await yachtsService.getRecommendations({ id });
  res.status(200).json(recommendations);
};

export const getSimilarYachtsById = async (req, res) => {
  const { id } = req.params;
  const recommendations = await yachtsService.getSimilarYachts(id);
  res.status(200).json(recommendations);
};
