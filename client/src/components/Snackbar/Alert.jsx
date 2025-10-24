import { forwardRef } from "react";

import { Alert as MuiAlert, Typography, SvgIcon } from "@mui/material";
import ExclamationIcon from "src/assets/icons/exclamation.svg";

export const Alert = forwardRef(({ variant = "error", message }, ref) => (
  <MuiAlert
    icon={<SvgIcon component={ExclamationIcon} />}
    ref={ref}
    variant="filled"
    severity={variant}
  >
    <Typography
      variant="medium"
      style={{ whiteSpace: "pre-line", wordBreak: "break-word" }}
    >
      {message}
    </Typography>
  </MuiAlert>
));

Alert.displayName = "SnackbarAlert";
