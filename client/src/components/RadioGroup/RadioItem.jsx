import MuiRadio from "@mui/material/Radio";
import { Stack, Typography, SvgIcon } from "@mui/material";
import RadioIcon from "src/assets/icons/radio.svg";
import RadioChecked from "src/assets/icons/radioChecked.svg";

const RadioItem = ({ label, value, checked, onCheck, ...props }) => (
  <Stack
    flexDirection="row"
    alignItems="center"
    style={{ cursor: "pointer" }}
    onClick={() => onCheck(value)}
  >
    <MuiRadio
      {...props}
      icon={<SvgIcon component={RadioIcon} />}
      checkedIcon={<SvgIcon component={RadioChecked} />}
      checked={checked}
      value={value}
    />
    <Typography variant="largeBold">{label}</Typography>
  </Stack>
);

export default RadioItem;
