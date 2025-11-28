import { useState } from "react";
import { useForm, FormProvider } from "react-hook-form";
import { useNavigate } from "react-router-dom";
import {
  Typography,
  Grid,
  Box,
  Alert,
  Button,
  Stack,
  IconButton
} from "@mui/material";
import { LoadingButton } from "@mui/lab";
import DeleteIcon from "@mui/icons-material/Delete";

import TextField from "src/components/TextField";
import { addYacht, getPresignedUrl, uploadFileToR2 } from "src/services/yachts";
import { ROUTES } from "src/navigation/routes";

const AddYachtPage = () => {
  const navigate = useNavigate();

  const [isUploading, setIsUploading] = useState(false);
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [uploadError, setUploadError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const methods = useForm({
    defaultValues: {
      name: "Mandarina",
      type: "Motor Yacht",
      guests: 11,
      cabins: 4,
      crew: 7,
      length: 29,
      year: 2020,
      model: "Custom",
      country: "Italy",
      baseMarina: "Amalfi Coast",
      description: "Some Description",
      summerLowSeasonPrice: 11000,
      summerHighSeasonPrice: 11000,
      winterLowSeasonPrice: 11000,
      winterHighSeasonPrice: 11000
    }
  });

  const { handleSubmit, register } = methods;

  const handleFileChange = (e) => {
    if (!e.target.files) return;

    const newFiles = Array.from(e.target.files);
    const newPreviews = newFiles.map((file) => URL.createObjectURL(file));

    setFiles((prev) => [...prev, ...newFiles]);
    setPreviews((prev) => [...prev, ...newPreviews]);

    e.target.value = "";
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    if (!droppedFiles.length) return;

    const droppedPreviews = droppedFiles.map((file) =>
      URL.createObjectURL(file)
    );

    setFiles((prev) => [...prev, ...droppedFiles]);
    setPreviews((prev) => [...prev, ...droppedPreviews]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const removeFile = (index) => {
    URL.revokeObjectURL(previews[index]);
    setFiles((prev) => prev.filter((_, i) => i !== index));
    setPreviews((prev) => prev.filter((_, i) => i !== index));
  };

  const onSubmit = async (data) => {
    setIsUploading(true);
    setUploadError(null);

    const upperName = data.name.trim().toUpperCase();

    try {
      const photoUrls = [];

      for (let i = 0; i < files.length; i++) {
        const file = files[i];

        const { uploadUrl, publicUrl } = await getPresignedUrl(
          upperName,
          i,
          file.type
        );

        await uploadFileToR2(uploadUrl, file);
        photoUrls.push(publicUrl);
      }

      const createdYacht = await addYacht({
        ...data,
        name: upperName,
        photos: photoUrls
      });

      if (createdYacht?.id) {
        navigate(ROUTES.YACHT_DETAILS.replace(":id", createdYacht.id));
      }
    } catch (error) {
      console.error(error);
      setUploadError("Saving error. Check your data and try again.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <FormProvider {...methods}>
      <Box
        component="form"
        onSubmit={handleSubmit(onSubmit)}
        sx={{ p: 4, maxWidth: 1200, mx: "auto" }}
      >
        <Typography variant="h4" mb={4}>
          Add new yacht
        </Typography>

        {uploadError && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {uploadError}
          </Alert>
        )}

        <Grid container spacing={6} wrap="nowrap">
          {/* LEFT - Upload */}
          <Grid item sx={{ width: "40%", flexShrink: 0 }}>
            <Box
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              sx={{
                border: "2px dashed #ccc",
                borderColor: isDragging ? "#000" : "#ccc",
                borderRadius: 3,
                height: 420,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexDirection: "column",
                textAlign: "center",
                cursor: "pointer",
                transition: "0.2s",
                backgroundColor: isDragging ? "#fafafa" : "#fff"
              }}
            >
              <Typography variant="body1" mb={1}>
                Drag & drop photos here
              </Typography>

              <Typography variant="body2" color="text.secondary" mb={2}>
                or
              </Typography>

              <Button variant="outlined" component="label">
                Select files
                <input
                  hidden
                  multiple
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                />
              </Button>
            </Box>

            {/* IMAGE PREVIEWS */}
            {files.length > 0 && (
              <Stack spacing={2} mt={3}>
                {/* MAIN PHOTO */}
                <Box
                  position="relative"
                  sx={{
                    borderRadius: 2,
                    overflow: "hidden",
                    border: "2px solid #1976d2"
                  }}
                >
                  <Box
                    component="img"
                    src={previews[0]}
                    alt="Main"
                    sx={{
                      width: "100%",
                      height: 250,
                      objectFit: "cover",
                      display: "block"
                    }}
                  />

                  <Stack
                    direction="row"
                    justifyContent="space-between"
                    alignItems="center"
                    sx={{
                      position: "absolute",
                      bottom: 0,
                      left: 0,
                      right: 0,
                      bgcolor: "rgba(0,0,0,0.6)",
                      p: 1
                    }}
                  >
                    <Typography variant="subtitle2" color="white" px={1}>
                      MAIN PHOTO
                    </Typography>

                    <IconButton
                      size="small"
                      onClick={() => removeFile(0)}
                      sx={{ color: "white" }}
                    >
                      <DeleteIcon />
                    </IconButton>
                  </Stack>
                </Box>

                {/* OTHER PHOTOS */}
                {files.length > 1 && (
                  <Grid container spacing={1}>
                    {files.map((file, index) => {
                      if (index === 0) return null;

                      return (
                        <Grid item xs={4} key={index}>
                          <Box
                            position="relative"
                            sx={{
                              borderRadius: 1,
                              overflow: "hidden",
                              height: 80
                            }}
                          >
                            <Box
                              component="img"
                              src={previews[index]}
                              sx={{
                                width: "100%",
                                height: "100%",
                                objectFit: "cover"
                              }}
                            />

                            <Box
                              onClick={() => removeFile(index)}
                              sx={{
                                position: "absolute",
                                inset: 0,
                                bgcolor: "rgba(0,0,0,0)",
                                transition: "0.2s",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                cursor: "pointer",
                                "&:hover": {
                                  bgcolor: "rgba(0,0,0,0.5)"
                                },
                                "&:hover .del-icon": {
                                  opacity: 1
                                }
                              }}
                            >
                              <DeleteIcon
                                className="del-icon"
                                sx={{ color: "white", opacity: 0 }}
                              />
                            </Box>
                          </Box>
                        </Grid>
                      );
                    })}
                  </Grid>
                )}
              </Stack>
            )}
          </Grid>

          {/* RIGHT - Form */}
          <Grid item sx={{ width: "60%" }}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <TextField label="Name" required {...register("name")} />
              </Grid>

              <Grid item xs={12}>
                <TextField label="Type" required {...register("type")} />
              </Grid>

              <Grid item xs={6}>
                <TextField
                  label="Guests"
                  isNumeric
                  {...register("guests", { valueAsNumber: true })}
                />
              </Grid>

              <Grid item xs={6}>
                <TextField
                  label="Cabins"
                  isNumeric
                  {...register("cabins", { valueAsNumber: true })}
                />
              </Grid>

              <Grid item xs={6}>
                <TextField
                  label="Crew"
                  isNumeric
                  {...register("crew", { valueAsNumber: true })}
                />
              </Grid>

              <Grid item xs={6}>
                <TextField
                  label="Length (m)"
                  isNumeric
                  {...register("length", { valueAsNumber: true })}
                />
              </Grid>

              <Grid item xs={6}>
                <TextField
                  label="Year"
                  isNumeric
                  {...register("year", { valueAsNumber: true })}
                />
              </Grid>

              <Grid item xs={6}>
                <TextField label="Model" {...register("model")} />
              </Grid>

              <Grid item xs={12}>
                <TextField label="Country" {...register("country")} />
              </Grid>

              <Grid item xs={12}>
                <TextField label="Base marina" {...register("baseMarina")} />
              </Grid>

              <Grid item xs={12}>
                <TextField
                  label="Description"
                  multiline
                  rows={4}
                  {...register("description")}
                />
              </Grid>

              <Grid item xs={12}>
                <Typography
                  variant="h6"
                  sx={{ mt: 2, mb: 1, fontWeight: "bold" }}
                >
                  Prices (per day)
                </Typography>

                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <TextField
                      label="Summer (Low)"
                      isNumeric
                      {...register("summerLowSeasonPrice", {
                        valueAsNumber: true
                      })}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      label="Summer (High)"
                      isNumeric
                      {...register("summerHighSeasonPrice", {
                        valueAsNumber: true
                      })}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      label="Winter (Low)"
                      isNumeric
                      {...register("winterLowSeasonPrice", {
                        valueAsNumber: true
                      })}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      label="Winter (High)"
                      isNumeric
                      {...register("winterHighSeasonPrice", {
                        valueAsNumber: true
                      })}
                    />
                  </Grid>
                </Grid>
              </Grid>

              <Grid item xs={12} mt={3}>
                <LoadingButton
                  loading={isUploading}
                  type="submit"
                  variant="contained"
                  size="large"
                  fullWidth
                  disabled={files.length === 0}
                >
                  Add yacht
                </LoadingButton>
              </Grid>
            </Grid>
          </Grid>
        </Grid>
      </Box>
    </FormProvider>
  );
};

export default AddYachtPage;
