import { IconButton, Stack } from "@mui/material";
import HeartIcon from "src/assets/icons/heart.svg";
import ProfileIcon from "src/assets/icons/profile.svg";
import BagIcon from "src/assets/icons/bag.svg";

const UserMenu = () => {
  // TODO add interaction with buttons
  return (
    <Stack
      direction={"row"}
      gap={1}
      alignItems={"center"}
      justifyContent={"space-between"}
    >
      <IconButton>
        <HeartIcon />
      </IconButton>
      <IconButton>
        <ProfileIcon />
      </IconButton>
      <IconButton>
        <BagIcon />
      </IconButton>
    </Stack>
  );
};

export default UserMenu;
