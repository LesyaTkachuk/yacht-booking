import {
  PasswordTextField,
  TextField,
  ModalFooter,
  showSuccess,
  showError,
} from "src/components";
import { FormProvider, useForm } from "react-hook-form";
import { Stack } from "@mui/material";
import * as yup from "yup";
import { yupResolver } from "@hookform/resolvers/yup";
import { loginUser } from "src/services/auth";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "src/context/authContext";

const schema = yup.object().shape({
  email: yup.string().email("Invalid email").required("Email is required"),
  password: yup.string().required("Password is required"),
});

const SignInForm = ({ onClose }) => {
  const queryClient = useQueryClient();
  const { setUser } = useAuth();
  const formMethods = useForm({
    defaultValues: { email: "", password: "" },
    resolver: yupResolver(schema),
  });

  const { register, handleSubmit } = formMethods;

  const { mutate: loginMutation, isPending } = useMutation({
    mutationFn: loginUser,
    onSuccess: (data) => {
      localStorage.setItem("token", data.token);
      localStorage.setItem("userId", data.user?.id);
      queryClient.refetchQueries(["currentUser"]);
      setUser({ ...data.user, token: data.token });
      onClose();
      showSuccess("User is successfully logged in");
    },

    onError: (e) => {
      showError(e, "There was an error logging in with provided credentials");
    },
  });

  const onLogin = (data) => {
    loginMutation(data);
  };

  return (
    <Stack gap={2} width={"76%"}>
      <FormProvider {...formMethods}>
        <TextField
          width="100%"
          label="Email"
          {...register("email")}
          tooltipTitle="Enter an email address to sign in"
        />
        <PasswordTextField
          width="100%"
          label="Password"
          {...register("password")}
          tooltipTitle="Enter a password"
        />
      </FormProvider>
      <ModalFooter
        onClose={onClose}
        onConfirm={handleSubmit(onLogin)}
        isPending={isPending}
        confirmBtnTitle="Sign In"
      />
    </Stack>
  );
};

export default SignInForm;
