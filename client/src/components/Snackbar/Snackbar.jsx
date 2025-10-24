"use client";

import { SnackbarProvider } from "notistack";

import { Alert } from "./Alert";

const Snackbar = () => (
  <SnackbarProvider
    maxSnack={3}
    autoHideDuration={5000}
    anchorOrigin={{
      vertical: "bottom",
      horizontal: "right",
    }}
    Components={{
      success: Alert,
      error: Alert,
      warning: Alert,
      info: Alert,
    }}
  />
);

export default Snackbar;
