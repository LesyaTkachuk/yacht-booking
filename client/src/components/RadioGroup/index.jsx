import { Controller, get } from "react-hook-form";

import { Stack, Typography } from "@mui/material";

import InfoIconButton from "../CustomIconButton";

import RadioItem from "./RadioItem";

const RadioGroup = ({
  formMethods,
  items,
  groupLabel,
  name,
  tooltipTitle,
  isRequired = true,
}) => {
  const {
    watch,
    control,
    formState: { errors },
  } = formMethods;
  const error = get(errors, name, false);

  const radioValue = watch(name);

  return (
    <Stack>
      <Stack direction="row" alignItems="center">
        <Typography variant="xMediumBold" color="secondary.contrastText">
          {groupLabel}
        </Typography>
        {tooltipTitle && <InfoIconButton tooltipTitle={tooltipTitle} />}
      </Stack>
      <Stack flexDirection="row" gap={3}>
        {items.map(({ value, title }) => (
          <Controller
            key={value}
            name={name}
            rules={{ required: isRequired && `${groupLabel} is required` }}
            control={control}
            render={({ field: { onChange } }) => (
              <RadioItem
                label={title}
                onCheck={onChange}
                value={value}
                checked={radioValue === value}
              />
            )}
          />
        ))}
      </Stack>
      {error?.message && (
        <Typography color="error.main" variant="medium" textAlign={"left"}>
          {error.message}
        </Typography>
      )}
    </Stack>
  );
};

export default RadioGroup;
