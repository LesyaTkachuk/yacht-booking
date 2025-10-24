import HttpError from "../helpers/HttpError.js";
import { verifyToken } from "../helpers/jwt.js";
import { findUser } from "../services/authService.js";
import {
  MISSING_AUTHORIZATION_HEADER,
  MISSING_BEARER,
  USER_NOT_FOUND,
  USER_NOT_AUTHORIZED,
} from "../constants/errorMessages.js";

const authenticate = async (req, _, next) => {
  try {
    const { authorization } = req.headers;
    if (!authorization) {
      throw HttpError(401, MISSING_AUTHORIZATION_HEADER);
    }

    const [bearer, token] = authorization.split(" ");
    if (bearer !== "Bearer") {
      throw HttpError(401, MISSING_BEARER);
    }

    const { payload, error } = verifyToken(token);
    if (error) {
      throw HttpError(401, error.message);
    }

    const user = await findUser({ id: payload.id });
    if (!user) {
      throw HttpError(401, USER_NOT_FOUND);
    }
    if (!user.token || user.token !== token) {
      throw HttpError(401, USER_NOT_AUTHORIZED);
    }
    req.user = user;
    next();
  } catch (error) {
    next(error);
  }
};

export default authenticate;
