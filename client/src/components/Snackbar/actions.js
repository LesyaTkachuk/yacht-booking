"use client";

import { useEffect } from "react";

import { STATUS_CODES } from "src/constants/errors";
import { transformErrorMessage } from "src/utils";
import { enqueueSnackbar } from "notistack";

export const showSuccess = (message) =>
  enqueueSnackbar(message, { variant: "success" });
export const showInfo = (message) =>
  enqueueSnackbar(message, { variant: "info" });
export const showWarning = (message) =>
  enqueueSnackbar(message, { variant: "warning" });

export const showError = (error, defaultMessage) => {
  let message = defaultMessage;
  let autoHideDuration = 5000;

  if (error) {
    // Network Error (user is offline or there is no response from server)
    // if (error?.code === ErrorCodes.NETWORK_ERROR) {
    //   message = error.message;
    // }

    // Validation Error - 422 status code (wrong data were provided)
    if (
      error?.status === STATUS_CODES.VALIDATION_ERROR &&
      error?.data?.message
    ) {
      message = transformErrorMessage(error.data.message);
      autoHideDuration = 10000;
    }

    // Bad Request - 400 status code, Conflict - 409
    if (
      [
        STATUS_CODES.BAD_REQUEST,
        STATUS_CODES.CONFLICT,
        STATUS_CODES.FORBIDDEN_RESOURCE,
        STATUS_CODES.NOT_FOUND,
        STATUS_CODES.UNAUTHORIZED,
      ].includes(error?.status) &&
      error?.data?.message
    ) {
      if (typeof error.data.message === "string") {
        message = error.data.message;
      }
      if (Array.isArray(error.data.message)) {
        message = transformErrorMessage(error.data.message);
      }
    }
  }
  return enqueueSnackbar(message, { variant: "error", autoHideDuration });
};

export const useShowError = (isError, message) => {
  useEffect(() => {
    if (isError && message) {
      showError(null, message);
    }
  }, [isError, message]);
};
