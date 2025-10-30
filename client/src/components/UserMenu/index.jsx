import { IconButton, Stack } from "@mui/material";
import HeartIcon from "src/assets/icons/heart.svg";
import ProfileIcon from "src/assets/icons/profile.svg";
import BagIcon from "src/assets/icons/bag.svg";
import { useModal } from "src/hooks";
import { UserModal } from "src/components";

const UserMenu = () => {
  const {
    isOpen: isUserModalOpen,
    open: openUserModal,
    close: closeUserModal,
  } = useModal();
  // TODO add interaction with buttons
  return (
    <>
      <Stack
        direction={"row"}
        gap={1}
        alignItems={"center"}
        justifyContent={"space-between"}
      >
        <IconButton>
          <HeartIcon />
        </IconButton>
        <IconButton onClick={openUserModal}>
          <ProfileIcon />
        </IconButton>
        <IconButton>
          <BagIcon />
        </IconButton>
      </Stack>
      {isUserModalOpen && <UserModal onClose={closeUserModal} />}
    </>
  );
};

export default UserMenu;
