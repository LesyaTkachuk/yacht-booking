import gravatar from "gravatar";
import path from "path";
import fs from "fs/promises";
import * as authService from "../services/authService.js";
import {
  LOGOUT_SUCCESS,
  VERIFY_SUCCESS,
  EMAIL_SENT,
} from "../constants/successMessages.js";
import { FILE_UPLOAD_ERROR } from "../constants/errorMessages.js";
import HttpError from "../helpers/HttpError.js";

export const signup = async (req, res) => {
  const avatarUrl = gravatar.url(req.body.email, {
    protocol: "https",
    s: "100",
  });

  const user = await authService.signup({ ...req.body, avatarUrl });

  res.status(201).json({
    user,
  });
};

export const verify = async (req, res) => {
  const { verificationToken } = req.params;

  await authService.verifyUser(verificationToken);

  res.status(200).json({ message: VERIFY_SUCCESS });
};

export const resendVerify = async (req, res) => {
  const { email } = req.body;

  await authService.resendVerifyEmail(email);

  res.status(200).json({ message: EMAIL_SENT });
};

export const signin = async (req, res) => {
  const result = await authService.signin(req.body);

  res.status(200).json(result);
};

export const getCurrent = async (req, res) => {
  res.status(200).json({
    id: req.user.id,
    email: req.user.email,
    role: req.user.role,
    avatarUrl: req.user.avatarUrl,
    ...(req.user.country && { country: req.user.country }),
    ...(req.user.sailingExp && { sailingExp: req.user.sailingExp }),
    ...(req.user.budgetMin && { budgetMin: req.user.budgetMin }),
    ...(req.user.budgetMax && { budgetMax: req.user.budgetMax }),
    ...(req.user.hasSkipperLicense && {
      hasSkipperLicense: req.user.hasSkipperLicense,
    }),
  });
};

// TODO update user role
export const updateUserRole = async (req, res) => {
  const user = await authService.updateUser({ id: req.user.id }, req.body);
  res.status(200).json(user);
};

export const updateUserAvatar = async (req, res) => {
  let avatarUrl = null;
  if (req.file) {
    try {
      const { path: oldPath, filename } = req.file;
      const avatarsDir = path.join(process.cwd(), "public", "avatars");
      const newPath = path.join(avatarsDir, filename);
      await fs.rename(oldPath, newPath);
      avatarUrl = path.join("avatars", filename);
    } catch (error) {
      await fs.unlink(req.file.path);
      throw HttpError(400, FILE_UPLOAD_ERROR);
    }
  }

  const user = await authService.updateUser({ id: req.user.id }, { avatarUrl });

  res
    .status(200)
    .json({ id: user.id, email: user.email, avatarUrl: user.avatarUrl });
};

export const signout = async (req, res) => {
  const { id } = req.user;

  await authService.logout({ id });

  res.status(200).json({ message: LOGOUT_SUCCESS });
};
