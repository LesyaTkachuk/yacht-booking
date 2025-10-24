import Joi from "joi";
import {
  EMAIL_REGEX,
  USER_ROLES,
  SAILING_EXPERIENCE,
} from "../constants/auth.js";

export const signupSchema = Joi.object({
  email: Joi.string()
    .pattern(EMAIL_REGEX)
    .message("Email is not valid")
    .required(),
  password: Joi.string().min(8).required(),
  role: Joi.string()
    .valid(USER_ROLES.LESSER, USER_ROLES.LESSEE, USER_ROLES.ADMIN)
    .required(),
  country: Joi.string().min(2).max(30).optional(),
  sailingExp: Joi.string()
    .valid(
      SAILING_EXPERIENCE.NONE,
      SAILING_EXPERIENCE.BEGINNER,
      SAILING_EXPERIENCE.INTERMEDIATE,
      SAILING_EXPERIENCE.PRO
    )
    .when("role", {
      is: USER_ROLES.LESSEE,
      then: Joi.required(),
      otherwise: Joi.forbidden(),
    }),
  budgetMin: Joi.number().when("role", {
    is: USER_ROLES.LESSEE,
    then: Joi.required(),
    otherwise: Joi.forbidden(),
  }),
  budgetMax: Joi.number().when("role", {
    is: USER_ROLES.LESSEE,
    then: Joi.required(),
    otherwise: Joi.forbidden(),
  }),
  hasSkipperLicense: Joi.boolean().when("role", {
    is: USER_ROLES.LESSEE,
    then: Joi.required(),
    otherwise: Joi.forbidden(),
  }),
});

export const verifySchema = Joi.object({
  email: Joi.string()
    .pattern(EMAIL_REGEX)
    .message("Email is not valid")
    .required(),
});

export const signinSchema = Joi.object({
  email: Joi.string()
    .pattern(EMAIL_REGEX)
    .message("Email is not valid")
    .required(),
  password: Joi.string().min(8).required(),
});

export const updateRoleSchema = Joi.object({
  role: Joi.string().valid(USER_ROLES.LESSER, USER_ROLES.LESSEE).required(),
});
