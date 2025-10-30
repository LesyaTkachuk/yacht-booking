import { Stack } from "@mui/material";

import CustomIconButton from "../CustomIconButton";

import { StyledTypography } from "./styled";

const SelectItem = ({
  icon,
  title = "",
  withTooltipTitle,
  tooltipTitle,
  ...props
}) => {
  return (
    <Stack direction="row" alignItems="center" gap={2} {...props}>
      {icon}
      <Stack direction="row" alignItems="center" gap={0.5}>
        <StyledTypography variant="body1">{title}</StyledTypography>
        {withTooltipTitle && tooltipTitle && (
          <CustomIconButton tooltipTitle={tooltipTitle} />
        )}
      </Stack>
    </Stack>
  );
};

export default SelectItem;
