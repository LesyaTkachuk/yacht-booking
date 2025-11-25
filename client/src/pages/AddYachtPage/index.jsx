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
import DeleteIcon from '@mui/icons-material/Delete';

import TextField from "src/components/TextField";
import { addYacht, getPresignedUrl, uploadFileToR2 } from "src/services/yachts";
import { ROUTES } from "src/navigation/routes";

const AddYachtPage = () => {
  const navigate = useNavigate();
  const [isUploading, setIsUploading] = useState(false);
  const [files, setFiles] = useState([]);
  const [uploadError, setUploadError] = useState(null);

  const methods = useForm({
    defaultValues: {
      name: "",
      type: "",
      guests: 0, 
      cabins: 0, 
      crew: 0, 
      length: 0, 
      year: 2024,
      model: "",
      country: "",
      baseMarina: "",
      description: "",
      summerLowSeasonPrice: 10000, 
      summerHighSeasonPrice: 10000,
      winterLowSeasonPrice: 10000, 
      winterHighSeasonPrice: 10000,
    }
  });

  const { handleSubmit, register } = methods;

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files);
      setFiles((prevFiles) => [...prevFiles, ...newFiles]);
    }
    e.target.value = ""; 
  };

  const removeFile = (indexToRemove) => {
    setFiles((prevFiles) => prevFiles.filter((_, index) => index !== indexToRemove));
  };

  const onSubmit = async (data) => {
    setIsUploading(true);
    setUploadError(null);

    const upperName = data.name.trim().toUpperCase();

    try {
      const photoUrls = [];

      if (files.length > 0) {
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
      }

      const createdYacht = await addYacht({
        ...data,
        name: upperName,
        photos: photoUrls,
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
        <Typography variant="h4" mb={4}>Add new yacht</Typography>
        
        {uploadError && <Alert severity="error" sx={{ mb: 3 }}>{uploadError}</Alert>}

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <TextField label="Name" required {...register("name", { required: "Name is required" })} />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField label="Type" required {...register("type", { required: "Type is required" })} />
          </Grid>
          
          <Grid item xs={6} md={3}>
            <TextField label="Guests" isNumeric required {...register("guests", { required: true, valueAsNumber: true })} />
          </Grid>
          <Grid item xs={6} md={3}>
            <TextField label="Cabins" isNumeric required {...register("cabins", { required: true, valueAsNumber: true })} />
          </Grid>
          <Grid item xs={6} md={3}>
            <TextField label="Crew" isNumeric {...register("crew", { valueAsNumber: true })} />
          </Grid>
          <Grid item xs={6} md={3}>
            <TextField label="Length (m)" isNumeric required {...register("length", { required: true, valueAsNumber: true })} />
          </Grid>
          
          <Grid item xs={12} md={4}>
            <TextField label="Year" isNumeric required {...register("year", { required: true, valueAsNumber: true })} />
          </Grid>
          <Grid item xs={12} md={4}>
            <TextField label="Model" {...register("model")} />
          </Grid>
          <Grid item xs={12} md={4}>
            <TextField label="Country" required {...register("country", { required: "Country is required" })} />
          </Grid>
          <Grid item xs={12}>
            <TextField label="Base marina" {...register("baseMarina")} />
          </Grid>
          
          <Grid item xs={12}>
            <Typography variant="h6" sx={{ mt: 2, mb: 1, fontWeight: 'bold' }}>
              Prices (per day)
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={6} md={3}>
                <TextField label="Summer (Low)" isNumeric required {...register("summerLowSeasonPrice", { required: true, valueAsNumber: true })} />
              </Grid>
              <Grid item xs={6} md={3}>
                <TextField label="Summer (High)" isNumeric required {...register("summerHighSeasonPrice", { required: true, valueAsNumber: true })} />
              </Grid>
              <Grid item xs={6} md={3}>
                <TextField label="Winter (Low)" isNumeric required {...register("winterLowSeasonPrice", { required: true, valueAsNumber: true })} />
              </Grid>
              <Grid item xs={6} md={3}>
                <TextField label="Winter (High)" isNumeric required {...register("winterHighSeasonPrice", { required: true, valueAsNumber: true })} />
              </Grid>
            </Grid>
          </Grid>

          <Grid item xs={12}>
             <TextField label="Description" multiline rows={4} {...register("description")} />
          </Grid>

          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Photos</Typography>
            <Typography variant="body2" color="text.secondary" mb={2}>
              First photo will be main.
            </Typography>
            
            <Button
              variant="outlined"
              component="label"
              sx={{ mb: 2 }}
            >
              Add photo
              <input 
                type="file" 
                hidden
                multiple 
                accept="image/*" 
                onChange={handleFileChange}
              />
            </Button>

            {files.length > 0 && (
              <Stack spacing={1} mb={3}>
                {files.map((file, index) => (
                  <Stack 
                    key={index} 
                    direction="row" 
                    alignItems="center" 
                    justifyContent="space-between"
                    sx={{ 
                      p: 1.5, 
                      border: '1px solid #e0e0e0', 
                      borderRadius: 1,
                      backgroundColor: index === 0 ? '#f5f9ff' : 'transparent'
                    }}
                  >
                    <Typography noWrap sx={{ maxWidth: '85%', fontSize: '0.9rem' }}>
                      {index === 0 ? <strong>[Main] </strong> : `${index + 1}. `}
                      {file.name}
                    </Typography>
                    <IconButton 
                      size="small" 
                      onClick={() => removeFile(index)} 
                      color="error"
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </Stack>
                ))}
              </Stack>
            )}
          </Grid>

          <Grid item xs={12} mt={2}>
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
      </Box>
    </FormProvider>
  );
};

export default AddYachtPage;
