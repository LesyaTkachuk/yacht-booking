import InfoOutlineIcon from "@mui/icons-material/InfoOutline";

import Tooltip from "../Tooltip";

import { StyledIconButton } from "./styled";

const CustomIconButton = ({
  tooltipTitle = "Info text",
  iconFontSize = "small",
  customIcon,
}) => {
  return (
    <Tooltip title={tooltipTitle}>
      <StyledIconButton disableRipple>
        {customIcon || (
          <InfoOutlineIcon color="primary" fontSize={iconFontSize} />
        )}
      </StyledIconButton>
    </Tooltip>
  );
};

export default CustomIconButton;
