import { CircularProgress } from "@mui/material";

import { StyledButton } from "./styled";

const Button = ({
  isLoading,
  children,
  size = "medium",
  variant = "contained",
  disabled,
  ...props
}) => (
  <StyledButton
    {...props}
    disabled={disabled || isLoading}
    size={size}
    variant={variant}
  >
    {size === "small" && isLoading ? null : children}
    {isLoading ? <CircularProgress size="18px" color="inherit" /> : null}
  </StyledButton>
);

export default Button;
