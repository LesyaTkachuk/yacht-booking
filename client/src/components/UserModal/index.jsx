import { Modal, Loader, ModalFooter, useShowError } from "src/components";
import { useQuery, useMutation } from "@tanstack/react-query";
import { getCurrentUser, logoutUser } from "src/services/auth";
import { useAuth } from "src/context/authContext";
import { capitalizeFirstLetter } from "src/utils";

import { Typography, Stack } from "@mui/material";

const UserModal = ({ onClose }) => {
  const { setUser } = useAuth();
  const {
    isPending,
    isError,
    data: user,
  } = useQuery({
    queryKey: ["currentUser"],
    queryFn: getCurrentUser,
  });

  const {
    email,
    role,
    country,
    sailingExp,
    budgetMin,
    budgetMax,
    hasSkipperLicense,
  } = user || {};

  useShowError(isError, "There was an error fetching user data");

  const { mutate: logoutMutation, isPending: isPendingLogout } = useMutation({
    mutationFn: logoutUser,
    onSuccess: () => {
      localStorage.removeItem("token");
      localStorage.removeItem("userId");
      setUser(null);
      onClose();
      showSuccess("User is successfully logged out");
    },

    onError: (e) => {
      showError(e, "There was an error logging in with provided credentials");
    },
  });

  const onLogout = async () => {
    await logoutMutation();
    localStorage.removeItem("token");
    setUser(null);
  };

  return (
    <Modal open title={"User Profile"} onClose={onClose}>
      {isPending && <Loader />}
      {!!user && (
        <Stack gap={2} marginY={4} alignItems={"flex-start"}>
          <Typography variant="body1">
            <strong>Email: </strong> {email}
          </Typography>
          <Typography variant="body1">
            <strong>Role: </strong> {capitalizeFirstLetter(role)}
          </Typography>
          {country && (
            <Typography variant="body1">
              <strong>Sailing Country: </strong> {country}
            </Typography>
          )}
          {sailingExp && (
            <Typography variant="body1">
              <strong>Sailing Experience: </strong>{" "}
              {capitalizeFirstLetter(sailingExp)}
            </Typography>
          )}
          {budgetMin && budgetMax && (
            <Typography variant="body1">
              <strong>Budget: </strong> {budgetMin}$ - {budgetMax}$
            </Typography>
          )}
          {hasSkipperLicense && (
            <Typography variant="body1">
              <strong>Sailing Experience: </strong>{" "}
              {hasSkipperLicense ? "Yes" : "No"}
            </Typography>
          )}
        </Stack>
      )}
      <ModalFooter
        onClose={onClose}
        onConfirm={onLogout}
        confirmBtnTitle="Logout"
        isPending={isPendingLogout}
      />
    </Modal>
  );
};

export default UserModal;
