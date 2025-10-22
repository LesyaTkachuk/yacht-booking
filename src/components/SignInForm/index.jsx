import { PasswordTextField, TextField, ModalFooter } from "src/components";
import { FormProvider, useForm } from "react-hook-form";
import { Stack } from "@mui/material";
import * as yup from "yup";
import { yupResolver } from "@hookform/resolvers/yup";

const schema = yup.object().shape({
  email: yup.string().email("Invalid email").required("Email is required"),
  password: yup.string().required("Password is required"),
});

const SignInForm = ({ onClose }) => {
  const formMethods = useForm({
    defaultValues: { email: "", password: "" },
    resolver: yupResolver(schema),
  });

  const { register, handleSubmit } = formMethods;

  //   TODO add login logic
  const onLogin = () => {};

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
      <ModalFooter onClose={onClose} onConfirm={handleSubmit(onLogin)} />
    </Stack>
  );
};

export default SignInForm;
