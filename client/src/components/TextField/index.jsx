"use client";

import { forwardRef } from "react";
import { get, useFormContext } from "react-hook-form";

import { TextField as MuiTextField, Stack, Typography } from "@mui/material";

import InfoIconButton from "../CustomIconButton";

const TextField = forwardRef(
  (
    { label, name, tooltipTitle, width, onChange, isNumeric = false, ...props },
    ref
  ) => {
    const {
      formState: { errors },
    } = useFormContext();
    const error = get(errors, name, false);

    const handleChange = (e) => {
      if (isNumeric) {
        const value = e.target.value.replace(/[^0-9.]/g, "");
        e.target.value = value;
      }

      onChange?.(e);
    };

    return (
      <Stack gap={1} flex={1} width={width ?? "inherit"}>
        {label && (
          <Stack direction="row" alignItems="center">
            <Typography variant="xMediumBold" color="secondary.contrastText">
              {label}
            </Typography>
            {tooltipTitle && <InfoIconButton tooltipTitle={tooltipTitle} />}
          </Stack>
        )}
        <MuiTextField
          error={!!error}
          inputRef={ref}
          {...props}
          onChange={handleChange}
          name={name}
          autoComplete="off"
        />
        {error?.message && (
          <Typography color="error.main" variant="medium" textAlign={"left"}>
            {error.message}
          </Typography>
        )}
      </Stack>
    );
  }
);

TextField.displayName = "TextField";

export default TextField;
